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

    in_dims_list = shard[0].all_shard_dims()
    out_dims_list = shard[2].all_shard_dims()
    in_dims = set(in_dims_list)
    out_dims = set(out_dims_list)
    assert len(in_dims ^ out_dims) == 1, "Can only split/gather over a single dimension"
    shard_dim = (in_dims ^ out_dims).pop()

    if in_dims < out_dims:
        # P * X ... -> X/P ...
        dim = out_dims_list.index(shard_dim)
        z = split_forward_allgather_backward(x, mesh[shard_dim].get_group(), dim, shapes)
    elif in_dims > out_dims:
        # X/P ... -> P * X ...
        dim = in_dims_list.index(shard_dim)
        z = allgather_forward_split_backward(x, mesh[shard_dim].get_group(), dim, shapes)
    else:
        assert False, "Cannot simultaneously gather and split"

    # permute if necessary
    return einsum(shard, z, name_only=True)
