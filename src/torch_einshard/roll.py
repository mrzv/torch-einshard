import torch

from .grammar import sharding
from .mappings import allgather_forward_split_backward, split_forward_allgather_backward


def _shape_for(shapes, shard_dim, axis_name):
    if not isinstance(shapes, dict):
        return shapes
    split_shapes = shapes.get(shard_dim)
    if isinstance(split_shapes, dict):
        return split_shapes.get(axis_name)
    return split_shapes


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

        split_shapes = _shape_for(shapes, axis.shard_dim, axis.name)
        group = mesh[axis.shard_dim].get_group()
        z = allgather_forward_split_backward(z, group, dim, split_shapes)
        z = torch.roll(z, shifts=shift, dims=dim)
        z = split_forward_allgather_backward(z, group, dim, split_shapes)

    return z
