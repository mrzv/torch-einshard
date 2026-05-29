import torch

from .grammar import parse_sharding
from .helpers import resolve_split_shapes
from .mappings import allgather_forward_split_backward, roll_shards_forward_backward, split_forward_allgather_backward


def einroll(shard, x, shifts, *, mesh=None, shapes=None):
    spec = parse_sharding(f"{shard} -> {shard}")[0]
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
        if split_shapes is not None and len(set(split_shapes)) == 1 and shift % split_shapes[0] == 0:
            z = roll_shards_forward_backward(z, group, shift // split_shapes[0])
            continue

        z = allgather_forward_split_backward(z, group, dim, split_shapes)
        z = torch.roll(z, shifts=shift, dims=dim)
        z = split_forward_allgather_backward(z, group, dim, split_shapes)

    return z
