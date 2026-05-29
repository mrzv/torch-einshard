import torch
from .helpers import all_reduce, all_gather, reduce_scatter, split

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
