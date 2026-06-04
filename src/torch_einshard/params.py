from dataclasses import dataclass, field

import torch.distributed as dist
import torch

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


def register_grad_reduction_hook_(ddp_model, mesh, ddp_group="dp"):
    def bucket_views(bucket):
        offset = 0
        views = []
        buffer = bucket.buffer()
        for param in bucket.parameters():
            view = buffer.narrow(0, offset, param.numel()).view_as(param)
            views.append((param, view))
            offset += param.numel()
        return views

    def reduction_hook(state, bucket):
        buffer = bucket.buffer()
        group = _group(mesh, ddp_group) if ddp_group is not None else None
        group_size = dist.get_world_size(group)

        if group_size == 1:
            future = torch.futures.Future()
            future.set_result(buffer)
        else:
            future = dist.all_reduce(buffer, group=group, async_op=True).get_future()

        def finish(future):
            if group_size > 1:
                buffer.div_(group_size)

            views = bucket_views(bucket)
            groups = sorted({
                name
                for param, _ in views
                for name in (get_param_spec(param).reduce if get_param_spec(param) is not None else ())
            })
            for name in groups:
                grad_views = [
                    view
                    for param, view in views
                    if get_param_spec(param) is not None and name in get_param_spec(param).reduce
                ]
                if not grad_views:
                    continue
                coalesced = torch.cat([view.reshape(-1) for view in grad_views])
                dist.all_reduce(coalesced, group=_group(mesh, name))
                offset = 0
                for view in grad_views:
                    view.copy_(coalesced.narrow(0, offset, view.numel()).view_as(view))
                    offset += view.numel()
            return buffer

        return future.then(finish)

    ddp_model.register_comm_hook(None, reduction_hook)
    return ddp_model
