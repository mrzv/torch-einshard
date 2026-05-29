import torch

from .grammar import sharding
from .helpers import resolve_split_shapes
from .mappings import allgather_forward_split_backward, split_forward_allgather_backward


def einroll(shard, x, shifts, *, mesh=None, shapes=None):
    spec = sharding(f"{shard} -> {shard}").map()[0]
    axes = spec.axes
    z = x

    for dim, axis in enumerate(axes):
        shift = shifts.get(axis.name, 0)
        if shift == 0:
            continue

        if axis.local():
            z = torch.roll(z, shifts=shift, dims=dim)
            continue

        group = mesh[axis.shard_dim].get_group()
        split_shapes = resolve_split_shapes(shapes, axis.shard_dim, axis.name, group)
        z = allgather_forward_split_backward(z, group, dim, split_shapes)
        z = torch.roll(z, shifts=shift, dims=dim)
        z = split_forward_allgather_backward(z, group, dim, split_shapes)

    return z
