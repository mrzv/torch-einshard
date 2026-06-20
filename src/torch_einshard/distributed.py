import warnings
import torch.distributed as dist

from .einsum import einsum
from .mappings import allreduce_forward_identity_backward, \
                       split_forward_allgather_backward, \
                       allgather_forward_split_backward, \
                        identity_forward_allreduce_backward, \
                        reducescatter_forward_allgather_backward, \
                        allgather_forward_reducescatter_backward, \
                        alltoall_repartition, \
                        owner_swap_forward_backward
from .helpers import resolve_split_shapes
from .sharding import Axis, Axes, EllipsisAxis, TensorSpec
from .symbolic import ExecutionPlan, build_unary_transition_plan, set_last_plan


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec


def _partials(spec):
    return getattr(spec, "partials", ())


def _without_ellipsis(axes):
    return Axes(axis for axis in axes if not isinstance(axis, EllipsisAxis))


def _expand_unary_ellipsis(input_axes, output_axes, x):
    input_has_ellipsis = any(isinstance(axis, EllipsisAxis) for axis in input_axes)
    output_has_ellipsis = any(isinstance(axis, EllipsisAxis) for axis in output_axes)
    if not input_has_ellipsis and not output_has_ellipsis:
        return input_axes, output_axes
    if input_has_ellipsis != output_has_ellipsis:
        raise NotImplementedError("Distributed unary ellipsis requires ellipsis in both input and output")

    input_width = x.dim() - (len(input_axes) - 1)
    output_width = x.dim() - (len(output_axes) - 1)
    if input_width < 0 or output_width < 0 or input_width != output_width:
        raise ValueError("Input tensor rank is incompatible with ellipsis notation")
    ellipsis_axes = Axes(Axis(f"__ellipsis{i}") for i in range(input_width))

    def expand(axes):
        expanded = Axes()
        for axis in axes:
            if isinstance(axis, EllipsisAxis):
                expanded.extend(ellipsis_axes)
            else:
                expanded.append(axis)
        return expanded

    return expand(input_axes), expand(output_axes)

def augment_parallelism(shard, mesh_dim_names):
    # TODO: add replication
    pass

def distributed_1d(shard, *xs, mesh, shapes):
    if mesh is not None:
        augment_parallelism(shard, mesh.mesh_dim_names)

    # X -> Z
    if shard[1] is None:
        return distributed_1d_1(shard, *xs, mesh, shapes)

    # X,Y -> Z
    elif shard[1] is not None:
        return distributed_1d_2(shard, *xs, mesh = mesh, shapes = shapes)
    else:
        return NotImplemented       # TODO: maybe raise instead?

def distributed_1d_2(shard, x, y, mesh, shapes = None):
    plan = ExecutionPlan()
    # TODO: check the dimensions match sharding

    input0_spec = shard[0]
    input1_spec = shard[1]
    output_spec = shard[2]

    def finish(output):
        set_last_plan(plan)
        return output

    input0_axes = _without_ellipsis(_axes(input0_spec))
    input1_axes = _without_ellipsis(_axes(input1_spec))
    output_axes = _without_ellipsis(_axes(output_spec))
    input0_by_name = {axis.name: axis for axis in input0_axes}
    input1_by_name = {axis.name: axis for axis in input1_axes}
    output_by_name = {axis.name: axis for axis in output_axes}
    output_names = {axis.name for axis in output_axes}
    contracted_names = [
        axis.name for axis in input0_axes
        if axis.name in input1_by_name and axis.name not in output_names
    ]

    contracted_target_by_name = {}
    for name in contracted_names:
        axis0 = input0_by_name[name]
        axis1 = input1_by_name[name]
        if axis0 == axis1:
            target = axis0
        elif axis0.local() and not axis1.local():
            target = axis1
        elif axis1.local() and not axis0.local():
            target = axis0
        elif axis0.shard_dim and axis1.shard_dim:
            target = axis0
        else:
            raise NotImplementedError("Unsupported contracted-axis sharding mismatch")
        contracted_target_by_name[name] = target

    reduction_dims = []
    for name in contracted_names:
        axis = contracted_target_by_name[name]
        if axis.local() or axis.shard_dim in reduction_dims:
            continue
        reduction_dims.append(axis.shard_dim)

    scatter_output_by_dim = {}
    for axis in output_axes:
        matching_input_axes = [
            input_axis for input_axis in (input0_by_name.get(axis.name), input1_by_name.get(axis.name))
            if input_axis is not None
        ]
        if axis.shard_dim in reduction_dims and any(input_axis.local() or input_axis == axis for input_axis in matching_input_axes):
            scatter_output_by_dim[axis.shard_dim] = axis

    def group(shard_dim):
        return mesh[shard_dim].get_group()

    def dim_of(axes, axis_name, tensor):
        dim = 0
        fixed_dims = sum(1 for axis in axes if not isinstance(axis, EllipsisAxis))
        ellipsis_dims = tensor.dim() - fixed_dims
        for axis in axes:
            if isinstance(axis, EllipsisAxis):
                dim += ellipsis_dims
            elif axis.name == axis_name:
                return dim
            else:
                dim += 1
        raise ValueError(f"Axis {axis_name!r} is not present in input")

    def active_in_output(shard_dim):
        if shard_dim in _partials(output_spec):
            return True
        return any(axis.shard_dim == shard_dim for axis in output_axes)

    def replace_axis(axes, axis_name, replacement):
        return Axes(
            replacement if not isinstance(axis, EllipsisAxis) and axis.name == axis_name else axis
            for axis in axes
        )

    def find_post_repartition():
        gather_candidates = []
        split_candidates = []
        for output_axis in output_axes:
            matching_input_axes = [
                input_axis for input_axis in (input0_by_name.get(output_axis.name), input1_by_name.get(output_axis.name))
                if input_axis is not None
            ]
            if len(matching_input_axes) != 1:
                continue
            input_axis = matching_input_axes[0]
            if input_axis == output_axis:
                continue
            if input_axis.shard_dim and output_axis.local():
                gather_candidates.append((input_axis, output_axis))
            elif input_axis.local() and output_axis.shard_dim:
                split_candidates.append((input_axis, output_axis))

        if len(gather_candidates) != 1 or len(split_candidates) != 1:
            return None

        source_input_axis, source_output_axis = gather_candidates[0]
        dest_input_axis, dest_output_axis = split_candidates[0]
        shard_dim = source_input_axis.shard_dim
        if shard_dim != dest_output_axis.shard_dim:
            return None
        if shard_dim in reduction_dims or shard_dim in _partials(output_spec):
            return None

        comm = group(shard_dim)
        source_shapes = resolve_split_shapes(shapes, shard_dim, source_input_axis.name, comm)
        dest_shapes = resolve_split_shapes(shapes, shard_dim, dest_output_axis.name, comm)
        if source_shapes is None or dest_shapes is None:
            return None

        return {
            "shard_dim": shard_dim,
            "source_input_axis": source_input_axis,
            "source_output_axis": source_output_axis,
            "dest_input_axis": dest_input_axis,
            "dest_output_axis": dest_output_axis,
            "source_shapes": source_shapes,
            "dest_shapes": dest_shapes,
        }

    post_repartition = find_post_repartition()

    def normalize_axis(tensor, normalized_axes, axis, target_axis):
        dim = dim_of(normalized_axes, axis.name, tensor)
        if axis.local():
            comm = group(target_axis.shard_dim)
            split_shapes = resolve_split_shapes(shapes, target_axis.shard_dim, axis.name, comm)
            tensor = plan.execute(
                "split_forward_allgather_backward",
                split_forward_allgather_backward,
                tensor,
                comm,
                dim,
                split_shapes,
                step_args=(axis.name, target_axis.shard_dim),
            )
        elif target_axis.local():
            comm = group(axis.shard_dim)
            split_shapes = resolve_split_shapes(shapes, axis.shard_dim, axis.name, comm)
            if active_in_output(axis.shard_dim):
                tensor = plan.execute(
                    "allgather_forward_reducescatter_backward",
                    allgather_forward_reducescatter_backward,
                    tensor,
                    comm,
                    dim,
                    split_shapes,
                    step_args=(axis.name, axis.shard_dim),
                )
            else:
                tensor = plan.execute(
                    "allgather_forward_split_backward",
                    allgather_forward_split_backward,
                    tensor,
                    comm,
                    dim,
                    split_shapes,
                    step_args=(axis.name, axis.shard_dim),
                )
        else:
            source_comm = group(axis.shard_dim)
            source_shapes = resolve_split_shapes(shapes, axis.shard_dim, axis.name, source_comm)
            if active_in_output(axis.shard_dim):
                tensor = plan.execute(
                    "allgather_forward_reducescatter_backward",
                    allgather_forward_reducescatter_backward,
                    tensor,
                    source_comm,
                    dim,
                    source_shapes,
                    step_args=(axis.name, axis.shard_dim),
                )
            else:
                tensor = plan.execute(
                    "allgather_forward_split_backward",
                    allgather_forward_split_backward,
                    tensor,
                    source_comm,
                    dim,
                    source_shapes,
                    step_args=(axis.name, axis.shard_dim),
                )

            dest_comm = group(target_axis.shard_dim)
            dest_shapes = resolve_split_shapes(shapes, target_axis.shard_dim, axis.name, dest_comm)
            tensor = plan.execute(
                "split_forward_allgather_backward",
                split_forward_allgather_backward,
                tensor,
                dest_comm,
                dim,
                dest_shapes,
                step_args=(axis.name, target_axis.shard_dim),
            )

        normalized_axes = replace_axis(normalized_axes, axis.name, target_axis)
        return tensor, normalized_axes

    def normalize_input(tensor, spec):
        axes = _axes(spec)
        normalized_axes = axes
        normalized_scatter_axes = set()
        owner_swapped_axes = set()

        # Crossed contracted axes must move ownership together; independent
        # gather/split steps can gather tensors with different logical slices.
        owner_swap_changes = []
        for axis in _without_ellipsis(axes):
            target_axis = contracted_target_by_name.get(axis.name)
            if target_axis is None or axis == target_axis:
                continue
            if axis.local() or target_axis.local():
                continue
            owner_swap_changes.append((axis, target_axis))

        if len(owner_swap_changes) >= 2:
            source_shard_dims = tuple(axis.shard_dim for axis, _ in owner_swap_changes)
            dest_shard_dims = tuple(target_axis.shard_dim for _, target_axis in owner_swap_changes)
            involved_shard_dims = set(source_shard_dims) | set(dest_shard_dims)
            changed_names = {axis.name for axis, _ in owner_swap_changes}
            can_owner_swap = (
                len(set(source_shard_dims)) == len(source_shard_dims)
                and len(set(dest_shard_dims)) == len(dest_shard_dims)
                and set(source_shard_dims) == set(dest_shard_dims)
            )
            can_owner_swap = can_owner_swap and all(
                axis.name in changed_names or axis.shard_dim not in involved_shard_dims
                for axis in _without_ellipsis(axes)
            )

            output_shape = list(tensor.shape)
            mesh_shape = mesh.mesh.shape
            mesh_dim_names = mesh.mesh_dim_names
            for axis, target_axis in owner_swap_changes:
                source_mesh_dim = mesh_dim_names.index(axis.shard_dim)
                dest_mesh_dim = mesh_dim_names.index(target_axis.shard_dim)
                if mesh_shape[source_mesh_dim] != mesh_shape[dest_mesh_dim]:
                    can_owner_swap = False
                    break

                source_comm = group(axis.shard_dim)
                dest_comm = group(target_axis.shard_dim)
                source_shapes = resolve_split_shapes(shapes, axis.shard_dim, axis.name, source_comm)
                dest_shapes = resolve_split_shapes(shapes, target_axis.shard_dim, axis.name, dest_comm)
                if source_shapes is None or dest_shapes is None or source_shapes != dest_shapes:
                    can_owner_swap = False
                    break
                output_shape[dim_of(normalized_axes, axis.name, tensor)] = dest_shapes[dist.get_rank(dest_comm)]

            if can_owner_swap:
                tensor = plan.execute(
                    "owner_swap",
                    owner_swap_forward_backward,
                    tensor,
                    mesh,
                    source_shard_dims,
                    dest_shard_dims,
                    tuple(output_shape),
                    step_args=(source_shard_dims, dest_shard_dims),
                )
                for axis, target_axis in owner_swap_changes:
                    normalized_axes = replace_axis(normalized_axes, axis.name, target_axis)
                    owner_swapped_axes.add(axis.name)
            else:
                raise NotImplementedError(
                    "Multiple contracted-axis shard-dimension changes require an owner-swap-compatible permutation"
                )

        # Gather reduce-scatter output axes before changing contracted axes;
        # otherwise a later gather can concatenate mismatched contracted slices.
        for axis in _without_ellipsis(axes):
            if axis.name in contracted_target_by_name or axis.name not in output_by_name:
                continue

            output_axis = output_by_name[axis.name]
            if scatter_output_by_dim.get(output_axis.shard_dim) != output_axis or axis != output_axis:
                continue

            dim = dim_of(normalized_axes, axis.name, tensor)
            comm = group(axis.shard_dim)
            split_shapes = resolve_split_shapes(shapes, axis.shard_dim, axis.name, comm)
            tensor = plan.execute(
                "allgather_forward_split_backward",
                allgather_forward_split_backward,
                tensor,
                comm,
                dim,
                split_shapes,
                step_args=(axis.name, axis.shard_dim),
            )
            normalized_axes = replace_axis(normalized_axes, axis.name, Axis(axis.name))
            normalized_scatter_axes.add(axis.name)

        for axis in _without_ellipsis(axes):
            if axis.name in normalized_scatter_axes:
                continue

            if axis.name in contracted_target_by_name:
                if axis.name in owner_swapped_axes:
                    continue
                target_axis = contracted_target_by_name[axis.name]
                if axis != target_axis:
                    tensor, normalized_axes = normalize_axis(tensor, normalized_axes, axis, target_axis)
                continue

            if post_repartition is not None:
                if axis == post_repartition["source_input_axis"]:
                    continue
                if axis == post_repartition["dest_input_axis"]:
                    tensor = plan.execute(
                        "identity_forward_allreduce_backward",
                        identity_forward_allreduce_backward,
                        tensor,
                        group(post_repartition["shard_dim"]),
                        step_args=(post_repartition["shard_dim"],),
                    )
                    continue

            if axis.name not in output_by_name:
                continue
            output_axis = output_by_name[axis.name]
            if axis == output_axis:
                continue
            if axis.local() and output_axis.shard_dim in reduction_dims:
                continue

            tensor, normalized_axes = normalize_axis(tensor, normalized_axes, axis, output_axis)

        return tensor, TensorSpec(normalized_axes, _partials(spec))

    def localize_scatter_axis(axis):
        if scatter_output_by_dim.get(axis.shard_dim) == axis:
            return Axis(axis.name)
        return axis

    def localize_post_repartition_axis(axis):
        if post_repartition is None:
            return axis
        if axis == post_repartition["source_output_axis"]:
            return post_repartition["source_input_axis"]
        if axis == post_repartition["dest_output_axis"]:
            return post_repartition["dest_input_axis"]
        return axis

    x, input0_spec = normalize_input(x, input0_spec)
    y, input1_spec = normalize_input(y, input1_spec)
    local_output_axes = Axes(
        localize_post_repartition_axis(localize_scatter_axis(axis)) if not isinstance(axis, EllipsisAxis) else axis
        for axis in _axes(output_spec)
    )
    local_output_spec = TensorSpec(local_output_axes, _partials(output_spec))
    normalized_shard = (input0_spec, input1_spec, local_output_spec)
    local_output_by_name = {axis.name: axis for axis in _without_ellipsis(local_output_axes)}

    input0_axes = _without_ellipsis(_axes(input0_spec))
    input1_axes = _without_ellipsis(_axes(input1_spec))
    input0_by_name = {axis.name: axis for axis in input0_axes}
    input1_by_name = {axis.name: axis for axis in input1_axes}
    shared_output_names = [
        axis.name for axis in input0_axes
        if axis.name in input1_by_name and axis.name in output_names
    ]

    for name in shared_output_names:
        axis = input0_by_name[name]
        assert axis == input1_by_name[name], "Shared axes must use matching sharding"
        assert axis == local_output_by_name[name], "Output shared axes must preserve input sharding"

    plan.add("rank_local_einsum")
    z = einsum(normalized_shard, x, y)    # perform the local operation
    current_output_axes = local_output_axes

    for shard_dim in reduction_dims:
        if shard_dim in _partials(shard[2]):
            scatter_axis = scatter_output_by_dim.get(shard_dim)
            if scatter_axis is not None:
                dim = dim_of(current_output_axes, scatter_axis.name, z)
                comm = group(shard_dim)
                split_shapes = resolve_split_shapes(shapes, shard_dim, scatter_axis.name, comm)
                z = plan.execute(
                    "split_forward_allgather_backward",
                    split_forward_allgather_backward,
                    z,
                    comm,
                    dim,
                    split_shapes,
                    step_args=(scatter_axis.name, shard_dim),
                )
                current_output_axes = replace_axis(current_output_axes, scatter_axis.name, scatter_axis)
            continue
        scatter_axis = scatter_output_by_dim.get(shard_dim)
        if scatter_axis is None:
            z = plan.execute(
                "allreduce_forward_identity_backward",
                allreduce_forward_identity_backward,
                z,
                comm=group(shard_dim),
                step_args=(shard_dim,),
            )
            continue

        dim = dim_of(current_output_axes, scatter_axis.name, z)
        comm = group(shard_dim)
        split_shapes = resolve_split_shapes(shapes, shard_dim, scatter_axis.name, comm)
        z = plan.execute(
            "reducescatter_forward_allgather_backward",
            reducescatter_forward_allgather_backward,
            z,
            comm,
            dim,
            split_shapes,
            step_args=(scatter_axis.name, shard_dim),
        )
        current_output_axes = replace_axis(current_output_axes, scatter_axis.name, scatter_axis)

    if post_repartition is not None:
        comm = group(post_repartition["shard_dim"])
        result = alltoall_repartition(
            z,
            comm,
            dim_of(current_output_axes, post_repartition["source_input_axis"].name, z),
            dim_of(current_output_axes, post_repartition["dest_input_axis"].name, z),
            post_repartition["source_shapes"],
            post_repartition["dest_shapes"],
        )
        if result is None:
            return None
        z = result
        plan.add(
            "alltoall_repartition",
            post_repartition["source_input_axis"].name,
            post_repartition["dest_input_axis"].name,
            post_repartition["shard_dim"],
        )
        current_output_axes = replace_axis(
            current_output_axes,
            post_repartition["source_output_axis"].name,
            post_repartition["source_output_axis"],
        )
        current_output_axes = replace_axis(
            current_output_axes,
            post_repartition["dest_output_axis"].name,
            post_repartition["dest_output_axis"],
        )

    return finish(z)

def distributed_1d_1(shard, x, mesh, shapes = None):
    plan = ExecutionPlan()
    input_spec = shard[0]
    output_spec = shard[2]
    input_axes = _axes(input_spec)
    output_axes = _axes(output_spec)
    input_axes, output_axes = _expand_unary_ellipsis(input_axes, output_axes, x)
    input_partials = list(_partials(input_spec))
    output_partials = list(_partials(output_spec))

    assert len(input_axes) == len(output_axes), "Input and output dimensions must match in the equation"
    if not (x.dim() == 0 and len(input_axes) == 1 and input_axes[0].name == output_axes[0].name):
        assert len(input_axes) == x.dim(), "Input dimensions must match those in the equation"

    in_by_name = {axis.name: axis for axis in input_axes}
    out_by_name = {axis.name: axis for axis in output_axes}
    assert in_by_name.keys() == out_by_name.keys(), "Input and output axes must match"
    transition_plan = build_unary_transition_plan(
        TensorSpec(input_axes, tuple(input_partials)),
        TensorSpec(output_axes, tuple(output_partials)),
    )

    z = x
    current = list(input_axes)
    current_partials = list(input_partials)

    def finish(output):
        set_last_plan(plan)
        return output

    shard_dim_changes = [
        out_axis.name for out_axis in output_axes
        if in_by_name[out_axis.name].shard_dim
        and out_axis.shard_dim
        and in_by_name[out_axis.name].shard_dim != out_axis.shard_dim
    ]

    def group(shard_dim):
        return mesh[shard_dim].get_group()

    def dim_of(axis_name):
        return next(i for i, axis in enumerate(current) if axis.name == axis_name)

    def execute_transition_step(step):
        nonlocal z, current
        if step.name == "allreduce_forward_identity_backward":
            partial, = step.args
            z = plan.execute(
                step.name,
                allreduce_forward_identity_backward,
                z,
                group(partial),
                step_args=step.args,
            )
            current_partials.remove(partial)
        elif step.name == "reducescatter_forward_allgather_backward":
            axis_name, partial = step.args
            dim = dim_of(axis_name)
            comm = group(partial)
            split_shapes = resolve_split_shapes(shapes, partial, axis_name, comm)
            z = plan.execute(
                step.name,
                reducescatter_forward_allgather_backward,
                z,
                comm,
                dim,
                split_shapes,
                step_args=step.args,
            )
            current[dim] = out_by_name[axis_name]
            current_partials.remove(partial)
        elif step.name == "split_forward_allgather_backward":
            axis_name, shard_dim = step.args
            dim = dim_of(axis_name)
            comm = group(shard_dim)
            split_shapes = resolve_split_shapes(shapes, shard_dim, axis_name, comm)
            z = plan.execute(
                step.name,
                split_forward_allgather_backward,
                z,
                comm,
                dim,
                split_shapes,
                step_args=step.args,
            )
            current[dim] = out_by_name[axis_name]
        elif step.name == "allgather_forward_split_backward":
            axis_name, shard_dim = step.args
            dim = dim_of(axis_name)
            comm = group(shard_dim)
            split_shapes = resolve_split_shapes(shapes, shard_dim, axis_name, comm)
            z = plan.execute(
                step.name,
                allgather_forward_split_backward,
                z,
                comm,
                dim,
                split_shapes,
                step_args=step.args,
            )
            current[dim] = Axis(axis_name)
        elif step.name == "allgather_forward_reducescatter_backward":
            axis_name, shard_dim = step.args
            dim = dim_of(axis_name)
            comm = group(shard_dim)
            split_shapes = resolve_split_shapes(shapes, shard_dim, axis_name, comm)
            z = plan.execute(
                step.name,
                allgather_forward_reducescatter_backward,
                z,
                comm,
                dim,
                split_shapes,
                step_args=step.args,
            )
            current[dim] = Axis(axis_name)
            current_partials.append(shard_dim)
        elif step.name == "identity_forward_allreduce_backward":
            partial, = step.args
            z = plan.execute(
                step.name,
                identity_forward_allreduce_backward,
                z,
                group(partial),
                step_args=step.args,
            )
            current_partials.append(partial)
        elif step.name == "permute":
            plan.add(step.name, *step.args)
            z = einsum(shard, z, name_only=True)
            current = list(output_axes)
        else:
            raise NotImplementedError(f"Unsupported unary plan step {step.name!r}")

    def has_repartition_change():
        gathered = 0
        split = 0
        changed_between_mesh_dims = False
        for out_axis in output_axes:
            in_axis = in_by_name[out_axis.name]
            if in_axis.shard_dim == out_axis.shard_dim:
                continue
            if in_axis.shard_dim and out_axis.shard_dim:
                changed_between_mesh_dims = True
            elif in_axis.shard_dim:
                gathered += 1
            elif out_axis.shard_dim:
                split += 1
        return changed_between_mesh_dims or (gathered > 0 and split > 0)

    def try_same_mesh_repartition():
        if output_partials or [axis.name for axis in input_axes] != [axis.name for axis in output_axes]:
            return None

        gather_axes = []
        split_axes = []
        for out_axis in output_axes:
            in_axis = in_by_name[out_axis.name]
            if in_axis.shard_dim == out_axis.shard_dim:
                continue
            if in_axis.shard_dim and out_axis.local():
                gather_axes.append(in_axis)
            elif in_axis.local() and out_axis.shard_dim:
                split_axes.append(out_axis)

        if len(gather_axes) != 1 or len(split_axes) != 1:
            return None
        source_axis = gather_axes[0]
        dest_axis = split_axes[0]
        if source_axis.shard_dim != dest_axis.shard_dim:
            return None

        comm = group(source_axis.shard_dim)
        source_shapes = resolve_split_shapes(shapes, source_axis.shard_dim, source_axis.name, comm)
        dest_shapes = resolve_split_shapes(shapes, dest_axis.shard_dim, dest_axis.name, comm)
        result = alltoall_repartition(
            z,
            comm,
            dim_of(source_axis.name),
            dim_of(dest_axis.name),
            source_shapes,
            dest_shapes,
        )
        if result is not None:
            plan.add("alltoall_repartition", source_axis.name, dest_axis.name, source_axis.shard_dim)
        return result

    def try_multi_axis_owner_swap():
        if output_partials or [axis.name for axis in input_axes] != [axis.name for axis in output_axes]:
            return None
        if len(shard_dim_changes) < 2:
            return None

        changed = [(in_by_name[name], out_by_name[name]) for name in shard_dim_changes]
        source_shard_dims = tuple(in_axis.shard_dim for in_axis, _ in changed)
        dest_shard_dims = tuple(out_axis.shard_dim for _, out_axis in changed)
        if set(source_shard_dims) != set(dest_shard_dims):
            return None

        mesh_shape = mesh.mesh.shape
        mesh_dim_names = mesh.mesh_dim_names
        for source_shard_dim, dest_shard_dim in zip(source_shard_dims, dest_shard_dims):
            source_mesh_dim = mesh_dim_names.index(source_shard_dim)
            dest_mesh_dim = mesh_dim_names.index(dest_shard_dim)
            if mesh_shape[source_mesh_dim] != mesh_shape[dest_mesh_dim]:
                return None

        output_shape = list(z.shape)
        for in_axis, out_axis in changed:
            source_comm = group(in_axis.shard_dim)
            dest_comm = group(out_axis.shard_dim)
            source_shapes = resolve_split_shapes(shapes, in_axis.shard_dim, in_axis.name, source_comm)
            dest_shapes = resolve_split_shapes(shapes, out_axis.shard_dim, out_axis.name, dest_comm)
            if source_shapes is None or dest_shapes is None or source_shapes != dest_shapes:
                return None
            output_shape[dim_of(out_axis.name)] = dest_shapes[dist.get_rank(dest_comm)]

        return plan.execute(
            "owner_swap",
            owner_swap_forward_backward,
            z,
            mesh,
            source_shard_dims,
            dest_shard_dims,
            tuple(output_shape),
            step_args=(source_shard_dims, dest_shard_dims),
        )

    remaining_steps = list(transition_plan.steps)
    while remaining_steps and remaining_steps[0].name in {
        "allreduce_forward_identity_backward",
        "reducescatter_forward_allgather_backward",
    }:
        execute_transition_step(remaining_steps.pop(0))

    optimized = try_multi_axis_owner_swap()
    if optimized is not None:
        z = optimized
        current = list(output_axes)
        if [axis.name for axis in current] == [axis.name for axis in output_axes]:
            return finish(z)

    assert len(shard_dim_changes) <= 1, "Cannot repartition multiple sharded axes between shard dimensions yet"

    optimized = try_same_mesh_repartition()
    if optimized is not None:
        z = optimized
        current = list(output_axes)
        if [axis.name for axis in current] == [axis.name for axis in output_axes]:
            return finish(z)

    if has_repartition_change():
        warnings.warn(
            "Using gather/split fallback for distributed repartition; this may materialize a larger tensor. "
            "The optimized path currently requires same-mesh axis-to-axis repartition with explicit split metadata.",
            RuntimeWarning,
            stacklevel=2,
        )

    for step in remaining_steps:
        execute_transition_step(step)

    if [axis.name for axis in current] == [axis.name for axis in output_axes]:
        return finish(z)

    raise AssertionError("Unary symbolic plan did not produce the requested output axes")
