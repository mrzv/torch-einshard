import torch
from .helpers import all_reduce, all_gather, all_to_all_repartition, reduce_scatter, roll_shards, split

class _AllReduceForwardIdentityBackward(torch.autograd.Function):
    """AllReduce in forward, Identity in backward"""

    @staticmethod
    def forward(ctx, input, comm):
        return all_reduce(input, group = comm)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class _IdentityForwardAllReduceBackward(torch.autograd.Function):
    """Identity in forward, AllReduce in backward"""

    @staticmethod
    def forward(ctx, input, comm):
        ctx.comm = comm
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return all_reduce(grad_output, group = ctx.comm), None

class _AllGatherForwardSplitBackward(torch.autograd.Function):
    """AllGather in forward and Split in backward"""

    @staticmethod
    def forward(ctx, input, comm, dim, shapes):
        ctx.comm = comm
        ctx.dim = dim
        ctx.shapes = shapes
        return all_gather(input, comm, dim, shapes)

    @staticmethod
    def backward(ctx, grad_output):
        return split(grad_output, ctx.comm, ctx.dim, ctx.shapes), None, None, None


class _SplitForwardAllGatherBackward(torch.autograd.Function):
    """Split in forward and AllGather in backward"""

    @staticmethod
    def forward(ctx, input, comm, dim, shapes):
        ctx.comm = comm
        ctx.dim = dim
        ctx.shapes = shapes
        return split(input, comm, dim, shapes)

    @staticmethod
    def backward(ctx, grad_output):
        return all_gather(grad_output, ctx.comm, ctx.dim, ctx.shapes), None, None, None

class _ReduceScatterForwardAllGatherBackward(torch.autograd.Function):
    """ReduceScatter in forward, AllGather in backward"""

    @staticmethod
    def forward(ctx, input, comm, dim, shapes):
        ctx.comm = comm
        ctx.dim = dim
        ctx.shapes = shapes
        return reduce_scatter(input, comm, dim, shapes)

    @staticmethod
    def backward(ctx, grad_output):
        return all_gather(grad_output, ctx.comm, ctx.dim, ctx.shapes), None, None, None

class _AllGatherForwardReduceScatterBackward(torch.autograd.Function):
    """AllGather in forward, ReduceScatter in backward"""

    @staticmethod
    def forward(ctx, input, comm, dim, shapes):
        ctx.comm = comm
        ctx.dim = dim
        ctx.shapes = shapes
        return all_gather(input, comm, dim, shapes)

    @staticmethod
    def backward(ctx, grad_output):
        return reduce_scatter(grad_output, ctx.comm, ctx.dim, ctx.shapes), None, None, None


class _AllToAllRepartition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, comm, source_dim, dest_dim, source_shapes, dest_shapes):
        ctx.comm = comm
        ctx.source_dim = source_dim
        ctx.dest_dim = dest_dim
        ctx.source_shapes = source_shapes
        ctx.dest_shapes = dest_shapes
        return all_to_all_repartition(input, comm, source_dim, dest_dim, source_shapes, dest_shapes)

    @staticmethod
    def backward(ctx, grad_output):
        return all_to_all_repartition(
            grad_output,
            ctx.comm,
            ctx.dest_dim,
            ctx.source_dim,
            ctx.dest_shapes,
            ctx.source_shapes,
        ), None, None, None, None, None


class _RollShards(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, comm, shard_shift):
        ctx.comm = comm
        ctx.shard_shift = shard_shift
        return roll_shards(input, comm, shard_shift)

    @staticmethod
    def backward(ctx, grad_output):
        return roll_shards(grad_output, ctx.comm, -ctx.shard_shift), None, None

def allreduce_forward_identity_backward(input, comm):
    return _AllReduceForwardIdentityBackward.apply(input, comm)

def identity_forward_allreduce_backward(input, comm):
    return _IdentityForwardAllReduceBackward.apply(input, comm)

def allgather_forward_split_backward(input, comm, dim, shapes):
    return _AllGatherForwardSplitBackward.apply(input, comm, dim, shapes)

def split_forward_allgather_backward(input, comm, dim, shapes):
    return _SplitForwardAllGatherBackward.apply(input, comm, dim, shapes)

def reducescatter_forward_allgather_backward(input, comm, dim, shapes):
    return _ReduceScatterForwardAllGatherBackward.apply(input, comm, dim, shapes)

def allgather_forward_reducescatter_backward(input, comm, dim, shapes):
    return _AllGatherForwardReduceScatterBackward.apply(input, comm, dim, shapes)

def alltoall_repartition(input, comm, source_dim, dest_dim, source_shapes, dest_shapes):
    return _AllToAllRepartition.apply(input, comm, source_dim, dest_dim, source_shapes, dest_shapes)

def roll_shards_forward_backward(input, comm, shard_shift):
    return _RollShards.apply(input, comm, shard_shift)
