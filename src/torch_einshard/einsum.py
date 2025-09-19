import string
import torch

# local contraction: translate the formula to einsum
def einsum(shard, *xs):
    shard_to_einsum = {}
    idx = 0
    equation = ""

    for i,n in enumerate(shard[0]):
        if n not in shard_to_einsum:
            shard_to_einsum[n] = string.ascii_lowercase[idx]
            idx += 1
        equation += shard_to_einsum[n]

    if shard[1] is not None:
        equation += ','
        for i,n in enumerate(shard[1]):
            if n not in shard_to_einsum:
                shard_to_einsum[n] = string.ascii_lowercase[idx]
                idx += 1
            equation += shard_to_einsum[n]
    equation += '->'
    for i,n in enumerate(shard[2]):
        assert n in shard_to_einsum, f"Output dimension {n} must be present in input"
        equation += shard_to_einsum[n]

    return torch.einsum(equation, *xs)
