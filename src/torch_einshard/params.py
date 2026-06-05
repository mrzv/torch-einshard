from dataclasses import dataclass, field

import torch.distributed as dist
import torch

from .grammar import parse_sharding
from .helpers import all_reduce, compute_split_shapes_for_factors
from .sharding import AxisGroup, EllipsisAxis


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


@dataclass(frozen=True)
class ParamShardMetadata:
    global_shape: tuple[int, ...]
    local_slices: tuple[slice, ...]
    local_shape: tuple[int, ...]
    shard_dims: tuple[str, ...]


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


def _require_spec(param_or_spec):
    if isinstance(param_or_spec, ParamSpec):
        return param_or_spec
    spec = get_param_spec(param_or_spec)
    if spec is None:
        raise ValueError("Parameter does not have an attached ParamSpec")
    return spec


def param_shard_dims(param_or_spec):
    spec = _require_spec(param_or_spec)
    return tuple(spec.axes.all_shard_dims())


def _mesh_rank_and_size(mesh, shard_dim):
    device_mesh = getattr(mesh, "device_mesh", mesh)
    if shard_dim in device_mesh.mesh_dim_names:
        mesh_dim = device_mesh.mesh_dim_names.index(shard_dim)
        rank = dist.get_rank()
        coord = (device_mesh.mesh == rank).nonzero()[0].tolist()
        return coord[mesh_dim], device_mesh.mesh.shape[mesh_dim]

    group = mesh[shard_dim].get_group()
    return dist.get_rank(group), dist.get_world_size(group)


def _axis_shard_dim(axis):
    if isinstance(axis, EllipsisAxis):
        raise ValueError("Parameter shard metadata does not support ellipsis axes")
    if isinstance(axis, AxisGroup):
        shard_dims = axis.axes.all_shard_dims()
        if shard_dims:
            raise NotImplementedError(
                "Parameter shard metadata does not support sharded factored-axis groups"
            )
        return ""
    return axis.shard_dim


def param_local_slices(param_or_spec, global_shape, mesh):
    spec = _require_spec(param_or_spec)
    global_shape = tuple(int(size) for size in global_shape)
    if len(global_shape) != len(spec.axes):
        raise ValueError(
            f"Global shape rank {len(global_shape)} does not match ParamSpec rank {len(spec.axes)}"
        )

    slices = []
    for axis, size in zip(spec.axes, global_shape):
        shard_dim = _axis_shard_dim(axis)
        if not shard_dim:
            slices.append(slice(None))
            continue

        rank, chunks = _mesh_rank_and_size(mesh, shard_dim)
        sections = compute_split_shapes_for_factors(size, chunks, 1)
        start = sum(sections[:rank])
        slices.append(slice(start, start + sections[rank]))
    return tuple(slices)


def param_local_shape(param_or_spec, global_shape, mesh):
    slices = param_local_slices(param_or_spec, global_shape, mesh)
    return _shape_from_slices(global_shape, slices)


def _shape_from_slices(global_shape, slices):
    shape = []
    for size, local_slice in zip(global_shape, slices):
        start = 0 if local_slice.start is None else local_slice.start
        stop = size if local_slice.stop is None else local_slice.stop
        shape.append(stop - start)
    return tuple(shape)


def param_shard_metadata(param_or_spec, global_shape, mesh):
    spec = _require_spec(param_or_spec)
    global_shape = tuple(int(size) for size in global_shape)
    local_slices = param_local_slices(spec, global_shape, mesh)
    return ParamShardMetadata(
        global_shape=global_shape,
        local_slices=local_slices,
        local_shape=_shape_from_slices(global_shape, local_slices),
        shard_dims=param_shard_dims(spec),
    )


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
