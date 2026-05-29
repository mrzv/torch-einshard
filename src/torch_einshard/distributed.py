from .einsum import einsum
from .mappings import allreduce_forward_identity_backward, \
                       split_forward_allgather_backward, \
                       allgather_forward_split_backward, \
                       identity_forward_allreduce_backward, \
                       reducescatter_forward_allgather_backward, \
                       allgather_forward_reducescatter_backward
from .helpers import resolve_split_shapes


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec


def _partials(spec):
    return getattr(spec, "partials", ())

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
        return distributed_1d_2(shard, *xs, mesh = mesh)
    else:
        return NotImplemented       # TODO: maybe raise instead?

def distributed_1d_2(shard, x, y, mesh):
    # TODO: check the dimensions match sharding

    input0_by_name = {axis.name: axis for axis in _axes(shard[0])}
    input1_by_name = {axis.name: axis for axis in _axes(shard[1])}
    output_names = {axis.name for axis in _axes(shard[2])}
    contracted_names = [
        axis.name for axis in _axes(shard[0])
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

    assert reduction_dims, "Expected distributed contraction, but got local"
    z = einsum(shard, x, y)    # perform the local operation

    for shard_dim in reduction_dims:
        if shard_dim in _partials(shard[2]):
            continue
        z = allreduce_forward_identity_backward(z, comm = mesh[shard_dim].get_group())

    return z

def distributed_1d_1(shard, x, mesh, shapes = None):
    input_spec = shard[0]
    output_spec = shard[2]
    input_axes = _axes(input_spec)
    output_axes = _axes(output_spec)
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

    def group(shard_dim):
        return mesh[shard_dim].get_group()

    def dim_of(axis_name):
        return next(i for i, axis in enumerate(current) if axis.name == axis_name)

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

    for out_axis in output_axes:
        in_axis = in_by_name[out_axis.name]
        current_axis = next(axis for axis in current if axis.name == out_axis.name)
        if current_axis.shard_dim == out_axis.shard_dim:
            continue
        assert current_axis.local() or out_axis.local(), "Cannot repartition between shard dimensions yet"

        shard_dim = current_axis.shard_dim or out_axis.shard_dim
        dim = dim_of(out_axis.name)
        comm = group(shard_dim)
        split_shapes = resolve_split_shapes(shapes, shard_dim, out_axis.name, comm)

        if current_axis.local():
            # P * X ... -> X/P ...
            z = split_forward_allgather_backward(z, comm, dim, split_shapes)
        else:
            # X/P ... -> P * X ...
            if shard_dim in output_partials:
                z = allgather_forward_reducescatter_backward(z, comm, dim, split_shapes)
                current_partials.append(shard_dim)
            else:
                z = allgather_forward_split_backward(z, comm, dim, split_shapes)

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
