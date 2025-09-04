from .einsum import einsum
from .mappings import allreduce_forward_identity_backward

def augment_parallelism(shard, mesh_dim_names):
    # TODO: add replication
    pass

def distributed_1d(shard, *xs, mesh):
    if mesh is not None:
        augment_parallelism(shard, mesh.mesh_dim_names)

    # X -> Z
    if xs[1] is None:
        return distributed_1d_1(shard, *xs, mesh, async_op)

    # X,Y -> Z
    elif xs[1] is not None:
        return distributed_1d_2(shard, *xs, mesh = mesh)
    else:
        return NotImplemented       # TODO: maybe raise instead?

def distributed_1d_2(shard, x, y, mesh):
    intersection = set(shard[0]) & set(shard[1])
    # TODO: this will not catch different sharding for the same dimension,
    #       in case we want to support that
    assert len(intersection) == 1, f"Only 1D contraction is supported, but intersection = {intersection}"
    common = intersection.pop()

    assert not common.local(), "Expected distributed contraction, but got local"
    # TODO: assert that shard[2] is replicated over common.shard_dim

    # ..., X / P, ..., X / P -> P * ...
    z = einsum(shard, x, y)    # perform the local operation

    # all_reduce z over p
    return allreduce_forward_identity_backward(z, comm = mesh[common.shard_dim].get_group())

def distributed_1d_1(shard, x, mesh, async_op):
    return NotImplemented
