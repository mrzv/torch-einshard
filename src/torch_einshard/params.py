from dataclasses import dataclass, field

import torch.distributed as dist

from .grammar import parse_sharding
from .helpers import all_reduce


PARAM_SPEC_ATTR = "einshard_spec"


def _tuple(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise TypeError("Parameter metadata entries must be strings")
    return result


def _parse_axes(spec):
    input_spec, other, output_spec = parse_sharding(f"{spec} -> {spec}")
    if other is not None or input_spec.partials or output_spec.partials:
        raise ValueError("Parameter specs must contain only axis layout notation")
    return input_spec.axes


@dataclass(frozen=True)
class ParamSpec:
    layout: str
    shared: tuple[str, ...] = ()
    reduce: tuple[str, ...] = ()
    axes: object = field(init=False)

    def __init__(self, layout, *, shared=(), reduce=()):
        axes = _parse_axes(layout)
        shared = _tuple(shared)
        reduce = _tuple(reduce)
        shard_dims = set(axes.all_shard_dims())
        for name in shared:
            overlap = shard_dims.intersection(name.split("-"))
            if overlap:
                dims = ", ".join(sorted(overlap))
                raise ValueError(f"Shared parameter metadata overlaps with sharded axis dimensions: {dims}")
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "shared", shared)
        object.__setattr__(self, "reduce", reduce)


def _group(mesh, name):
    return mesh[name].get_group()


def sync_param_(param, spec, mesh):
    for name in spec.shared:
        group = _group(mesh, name)
        if dist.get_world_size(group) == 1:
            continue
        dist.broadcast(param.data, src=dist.get_global_rank(group, 0), group=group)
    return param


def reduce_grad_(param, spec, mesh):
    if param.grad is None:
        return param
    for name in spec.reduce:
        param.grad = all_reduce(param.grad, _group(mesh, name))
    return param


def set_param_spec(param, spec):
    setattr(param, PARAM_SPEC_ATTR, spec)
    return param


def get_param_spec(param):
    return getattr(param, PARAM_SPEC_ATTR, None)


def sync_module_params_(module, mesh):
    for param in module.parameters():
        spec = get_param_spec(param)
        if spec is not None:
            sync_param_(param, spec, mesh)
    return module


def reduce_module_grads_(module, mesh):
    for param in module.parameters():
        spec = get_param_spec(param)
        if spec is not None:
            reduce_grad_(param, spec, mesh)
    return module
