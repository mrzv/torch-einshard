import torch
import warnings

from .grammar import parse_sharding
from .families import cached_expand_axis_families, expand_family_mapping
from .helpers import resolve_split_shapes
from .mappings import allgather_forward_split_backward, roll_sharded_forward_backward, split_forward_allgather_backward


def einroll(shard, x, shifts, *, mesh=None, shapes=None, families=None):
    shard, _ = cached_expand_axis_families(shard, families=families)
    shifts = expand_family_mapping(shifts, families, label="Shift")
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
        if split_shapes is not None:
            z = roll_sharded_forward_backward(z, group, dim, shift, split_shapes)
            continue

        warnings.warn(
            f"Using gather/roll/split fallback for sharded roll axis {axis.name!r}; "
            "provide shapes to enable direct slice exchange.",
            RuntimeWarning,
            stacklevel=2,
        )
        z = allgather_forward_split_backward(z, group, dim, split_shapes)
        z = torch.roll(z, shifts=shift, dims=dim)
        z = split_forward_allgather_backward(z, group, dim, split_shapes)

    return z
