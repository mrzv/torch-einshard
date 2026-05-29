from .einsum import einsum
from .mappings import allreduce_forward_identity_backward, \
                      split_forward_allgather_backward, \
                      allgather_forward_split_backward

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

    intersection = set(shard[0]) & set(shard[1])
    # TODO: this will not catch different sharding for the same dimension,
    #       in case we want to support that
    assert len(intersection) == 1, f"Only 1D contraction is supported, but intersection = {intersection}"
    common = intersection.pop()

    assert not common.local(), "Expected distributed contraction, but got local"
    # TODO: assert that shard[2] is replicated over common.shard_dim

    # ... X/P, ... X/P -> P * ...
    z = einsum(shard, x, y)    # perform the local operation

    # all_reduce z over p
    return allreduce_forward_identity_backward(z, comm = mesh[common.shard_dim].get_group())

def distributed_1d_1(shard, x, mesh, shapes = None):
    assert len(shard[0]) == len(shard[2]), "Input and output dimensions must match in the equation"
    assert len(shard[0]) == x.dim(), "Input dimensions must match those in the equation"

    in_by_name = {axis.name: axis for axis in shard[0]}
    out_by_name = {axis.name: axis for axis in shard[2]}
    assert in_by_name.keys() == out_by_name.keys(), "Input and output axes must match"

    z = x
    current = list(shard[0])

    def shapes_for(shard_dim, axis_name):
        if not isinstance(shapes, dict):
            return shapes
        split_shapes = shapes.get(shard_dim)
        if isinstance(split_shapes, dict):
            return split_shapes.get(axis_name)
        return split_shapes

    for out_axis in shard[2]:
        in_axis = in_by_name[out_axis.name]
        if in_axis.shard_dim == out_axis.shard_dim:
            continue
        assert in_axis.local() or out_axis.local(), "Cannot repartition between shard dimensions yet"

        shard_dim = in_axis.shard_dim or out_axis.shard_dim
        dim = next(i for i, axis in enumerate(current) if axis.name == out_axis.name)
        split_shapes = shapes_for(shard_dim, out_axis.name)

        if in_axis.local():
            # P * X ... -> X/P ...
            z = split_forward_allgather_backward(z, mesh[shard_dim].get_group(), dim, split_shapes)
        else:
            # X/P ... -> P * X ...
            z = allgather_forward_split_backward(z, mesh[shard_dim].get_group(), dim, split_shapes)

        current[dim] = out_axis

    # permute if necessary
    return einsum(shard, z, name_only=True)
