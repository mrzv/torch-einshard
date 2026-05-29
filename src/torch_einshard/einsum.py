import string
import torch


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec

# local contraction: translate the formula to einsum
def einsum(shard, *xs, name_only = False):
    if not name_only:
        assert not getattr(shard[0], "partials", ()), "Local einsum does not support partial inputs"
        if shard[1] is not None:
            assert not getattr(shard[1], "partials", ()), "Local einsum does not support partial inputs"

    shard_to_einsum = {}
    idx = 0
    equation = ""

    for i,n in enumerate(_axes(shard[0])):
        if name_only:
            n = n.name
        if n not in shard_to_einsum:
            shard_to_einsum[n] = string.ascii_lowercase[idx]
            idx += 1
        equation += shard_to_einsum[n]

    if shard[1] is not None:
        equation += ','
        for i,n in enumerate(_axes(shard[1])):
            if name_only:
                n = n.name
            if n not in shard_to_einsum:
                shard_to_einsum[n] = string.ascii_lowercase[idx]
                idx += 1
            equation += shard_to_einsum[n]
    equation += '->'
    for i,n in enumerate(_axes(shard[2])):
        if name_only:
            n = n.name
        assert n in shard_to_einsum, f"Output dimension {n} must be present in input"
        equation += shard_to_einsum[n]

    return torch.einsum(equation, *xs)
