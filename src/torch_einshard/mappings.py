import torch
from .helpers import all_reduce

class _AllReduceForwardIdentityBackward(torch.autograd.Function):
    """AllReduce in forward, Identity in backward"""

    @staticmethod
    def forward(ctx, input, comm):
        return all_reduce(input, group = comm)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def allreduce_forward_identity_backward(input, comm):
    return _AllReduceForwardIdentityBackward.apply(input, comm)

