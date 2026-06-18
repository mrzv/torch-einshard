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
    # TODO: check the dimensions match sharding

    input0_spec = shard[0]
    input1_spec = shard[1]
    output_spec = shard[2]
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
    assert contracted_names, "Expected a contraction axis"

    reduction_dims = []
    for name in contracted_names:
        axis = input0_by_name[name]
        assert axis == input1_by_name[name], "Contracted axes must use matching sharding"
        if axis.local() or axis.shard_dim in reduction_dims:
            continue
        reduction_dims.append(axis.shard_dim)

    scatter_output_by_dim = {}
    for axis in output_axes:
        matching_input_axes = [
            input_axis for input_axis in (input0_by_name.get(axis.name), input1_by_name.get(axis.name))
            if input_axis is not None
        ]
        if axis.shard_dim in reduction_dims and any(input_axis.local() for input_axis in matching_input_axes):
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

    def normalize_input(tensor, spec):
        axes = _axes(spec)
        normalized_axes = axes
        for axis in _without_ellipsis(axes):
            if axis.name not in output_by_name:
                continue
            output_axis = output_by_name[axis.name]
            if axis == output_axis:
                continue
            if axis.local() and output_axis.shard_dim in reduction_dims:
                continue

            dim = dim_of(normalized_axes, axis.name, tensor)
            if axis.local():
                comm = group(output_axis.shard_dim)
                split_shapes = resolve_split_shapes(shapes, output_axis.shard_dim, axis.name, comm)
                tensor = split_forward_allgather_backward(tensor, comm, dim, split_shapes)
            elif output_axis.local():
                comm = group(axis.shard_dim)
                split_shapes = resolve_split_shapes(shapes, axis.shard_dim, axis.name, comm)
                if active_in_output(axis.shard_dim):
                    tensor = allgather_forward_reducescatter_backward(tensor, comm, dim, split_shapes)
                else:
                    tensor = allgather_forward_split_backward(tensor, comm, dim, split_shapes)
            else:
                source_comm = group(axis.shard_dim)
                source_shapes = resolve_split_shapes(shapes, axis.shard_dim, axis.name, source_comm)
                if active_in_output(axis.shard_dim):
                    tensor = allgather_forward_reducescatter_backward(tensor, source_comm, dim, source_shapes)
                else:
                    tensor = allgather_forward_split_backward(tensor, source_comm, dim, source_shapes)

                dest_comm = group(output_axis.shard_dim)
                dest_shapes = resolve_split_shapes(shapes, output_axis.shard_dim, axis.name, dest_comm)
                tensor = split_forward_allgather_backward(tensor, dest_comm, dim, dest_shapes)

            normalized_axes = replace_axis(normalized_axes, axis.name, output_axis)

        return tensor, TensorSpec(normalized_axes, _partials(spec))

    def localize_scatter_axis(axis):
        if scatter_output_by_dim.get(axis.shard_dim) == axis:
            return Axis(axis.name)
        return axis

    x, input0_spec = normalize_input(x, input0_spec)
    y, input1_spec = normalize_input(y, input1_spec)
    local_output_axes = Axes(
        localize_scatter_axis(axis) if not isinstance(axis, EllipsisAxis) else axis
        for axis in _axes(output_spec)
    )
    local_output_spec = TensorSpec(local_output_axes, _partials(output_spec))
    normalized_shard = (input0_spec, input1_spec, local_output_spec)

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
        assert axis == output_by_name[name], "Output shared axes must preserve input sharding"

    z = einsum(normalized_shard, x, y)    # perform the local operation
    current_output_axes = local_output_axes

    for shard_dim in reduction_dims:
        if shard_dim in _partials(shard[2]):
            continue
        scatter_axis = scatter_output_by_dim.get(shard_dim)
        if scatter_axis is None:
            z = allreduce_forward_identity_backward(z, comm = group(shard_dim))
            continue

        dim = dim_of(current_output_axes, scatter_axis.name, z)
        comm = group(shard_dim)
        split_shapes = resolve_split_shapes(shapes, shard_dim, scatter_axis.name, comm)
        z = reducescatter_forward_allgather_backward(z, comm, dim, split_shapes)
        current_output_axes = replace_axis(current_output_axes, scatter_axis.name, scatter_axis)

    return z

def distributed_1d_1(shard, x, mesh, shapes = None):
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

    z = x
    current = list(input_axes)
    current_partials = list(input_partials)

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
        return alltoall_repartition(
            z,
            comm,
            dim_of(source_axis.name),
            dim_of(dest_axis.name),
            source_shapes,
            dest_shapes,
        )

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

        return owner_swap_forward_backward(
            z,
            mesh,
            source_shard_dims,
            dest_shard_dims,
            tuple(output_shape),
        )

    # Reduce input partials first. If the same mesh dimension appears on an
    # output axis, reduce-scatter directly into that shard; otherwise all-reduce.
    for partial in list(current_partials):
        if partial in output_partials:
            continue

        scatter_axis = None
        for out_axis in output_axes:
            in_axis = next(axis for axis in current if axis.name == out_axis.name)
            if in_axis.local() and out_axis.shard_dim == partial:
                scatter_axis = out_axis
                break

        if scatter_axis is None:
            z = allreduce_forward_identity_backward(z, group(partial))
        else:
            dim = dim_of(scatter_axis.name)
            comm = group(partial)
            split_shapes = resolve_split_shapes(shapes, partial, scatter_axis.name, comm)
            z = reducescatter_forward_allgather_backward(z, comm, dim, split_shapes)
            current[dim] = scatter_axis
        current_partials.remove(partial)

    optimized = try_multi_axis_owner_swap()
    if optimized is not None:
        z = optimized
        current = list(output_axes)
        if [axis.name for axis in current] == [axis.name for axis in output_axes]:
            return z

    assert len(shard_dim_changes) <= 1, "Cannot repartition multiple sharded axes between shard dimensions yet"

    optimized = try_same_mesh_repartition()
    if optimized is not None:
        z = optimized
        current = list(output_axes)
        if [axis.name for axis in current] == [axis.name for axis in output_axes]:
            return z

    if has_repartition_change():
        warnings.warn(
            "Using gather/split fallback for distributed repartition; this may materialize a larger tensor. "
            "The optimized path currently requires same-mesh axis-to-axis repartition with explicit split metadata.",
            RuntimeWarning,
            stacklevel=2,
        )

    for out_axis in output_axes:
        in_axis = in_by_name[out_axis.name]
        current_axis = next(axis for axis in current if axis.name == out_axis.name)
        if current_axis.shard_dim == out_axis.shard_dim:
            continue

        dim = dim_of(out_axis.name)

        if current_axis.local():
            # P * X ... -> X/P ...
            shard_dim = out_axis.shard_dim
            comm = group(shard_dim)
            split_shapes = resolve_split_shapes(shapes, shard_dim, out_axis.name, comm)
            z = split_forward_allgather_backward(z, comm, dim, split_shapes)
        elif out_axis.local():
            # X/P ... -> P * X ...
            shard_dim = current_axis.shard_dim
            comm = group(shard_dim)
            split_shapes = resolve_split_shapes(shapes, shard_dim, out_axis.name, comm)
            if shard_dim in output_partials:
                z = allgather_forward_reducescatter_backward(z, comm, dim, split_shapes)
                current_partials.append(shard_dim)
            else:
                z = allgather_forward_split_backward(z, comm, dim, split_shapes)
        else:
            # X/P ... -> Q * X ... -> X/Q ...
            source_comm = group(current_axis.shard_dim)
            source_shapes = resolve_split_shapes(shapes, current_axis.shard_dim, out_axis.name, source_comm)
            z = allgather_forward_split_backward(z, source_comm, dim, source_shapes)

            dest_comm = group(out_axis.shard_dim)
            dest_shapes = resolve_split_shapes(shapes, out_axis.shard_dim, out_axis.name, dest_comm)
            z = split_forward_allgather_backward(z, dest_comm, dim, dest_shapes)

        current[dim] = out_axis

    for partial in output_partials:
        if partial in current_partials:
            continue
        z = identity_forward_allreduce_backward(z, group(partial))
        current_partials.append(partial)

    if [axis.name for axis in current] == [axis.name for axis in output_axes]:
        return z

    # permute if necessary
    return einsum(shard, z, name_only=True)
