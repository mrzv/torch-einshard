from dataclasses import dataclass, field, replace

import torch.distributed as dist
import torch

from .grammar import parse_sharding
from .helpers import all_reduce, compute_split_shapes_for_factors
from .sharding import AxisGroup, EllipsisAxis, TensorSpec
from .symbolic import TensorState


PARAM_SPEC_ATTR = "einshard_spec"
PARAM_STATE_ATTR = "einshard_state"


def _tuple(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise TypeError("Parameter metadata entries must be strings")
    return result


def _duplicates(values):
    seen = set()
    result = []
    for value in values:
        if value in seen and value not in result:
            result.append(value)
        seen.add(value)
    return result


def _mesh_dim_components(name):
    return {name, *name.split("-")}


def _mesh_dims_components(names):
    result = set()
    for name in names:
        result.update(_mesh_dim_components(name))
    return result


def _validate_parameter_layout(layout_shard_dims, init_sync=None):
    seen_components = set()
    for shard_dim in layout_shard_dims:
        components = _mesh_dim_components(shard_dim)
        overlap = seen_components.intersection(components)
        if overlap:
            dims = ", ".join(sorted(overlap))
            raise ValueError(f"Parameter specs cannot shard multiple axes over the same mesh dimension: {dims}")
        seen_components.update(components)

    if init_sync is None or init_sync.mode != "explicit":
        return

    shard_components = _mesh_dims_components(layout_shard_dims)
    for name in init_sync.mesh_dims:
        overlap = shard_components.intersection(_mesh_dim_components(name))
        if overlap:
            dims = ", ".join(sorted(overlap))
            raise ValueError(f"Parameter init_sync annotation overlaps with sharded axis dimensions: {dims}")


def _parse_axes(spec):
    input_spec, other, output_spec = parse_sharding(f"{spec} -> {spec}")
    if other is not None or input_spec.partials or output_spec.partials:
        raise ValueError("Parameter specs must contain only axis layout notation")
    return input_spec.axes


def _tensor_state_or_none(spec, mesh_dim_names):
    try:
        return TensorState.from_spec(spec, mesh_dim_names=mesh_dim_names)
    except ValueError:
        return None


def _flat_axes(spec):
    axes = spec.axes if hasattr(spec, "axes") else spec
    return axes.flat() if hasattr(axes, "flat") else axes


def _axis_signature(axis):
    if isinstance(axis, EllipsisAxis):
        return ("ellipsis",)
    if isinstance(axis, AxisGroup):
        return ("group", tuple(_axis_signature(inner) for inner in axis.axes))
    return ("axis", axis.name, axis.shard_dim)


def _spec_layout_signature(spec):
    return tuple(_axis_signature(axis) for axis in spec.axes)


def _add_unique_mesh_dim(mesh_dims, name):
    if name and name not in mesh_dims:
        mesh_dims.append(name)


@dataclass(frozen=True)
class ParameterInitSync:
    mode: str = "none"
    mesh_dims: tuple[str, ...] = ()

    @classmethod
    def from_annotation(cls, annotation, layout_shard_dims, mesh_dim_names):
        if annotation is not None:
            if annotation.mode in {"none", "external"}:
                return cls(mode=annotation.mode)
            if annotation.mode == "explicit":
                return cls(mode="explicit", mesh_dims=annotation.mesh_dims)

        if not mesh_dim_names:
            return cls(mode="none")
        shard_dims = _mesh_dims_components(layout_shard_dims)
        mesh_dims = tuple(
            dim for dim in mesh_dim_names
            if not _mesh_dim_components(dim).intersection(shard_dims)
        )
        return cls(mode="inferred", mesh_dims=mesh_dims)


@dataclass(frozen=True)
class ParameterGradComm:
    mode: str = "none"
    mesh_dims: tuple[str, ...] = ()
    backend: str = "none"
    schedule: str = "backend_default"

    @classmethod
    def from_annotation(cls, annotation, *, is_param=False):
        if annotation is None:
            if is_param:
                return cls(mode="inferred", backend="native")
            return cls()
        return cls(
            mode=annotation.mode,
            mesh_dims=annotation.mesh_dims,
            backend=annotation.backend,
            schedule=annotation.schedule,
        )

    @classmethod
    def from_reduce_groups(cls, reduce):
        reduce = _tuple(reduce)
        if not reduce:
            return cls()
        return cls(mode="explicit", mesh_dims=reduce, backend="native", schedule="synchronous")

    @property
    def pending_inference(self):
        return self.mode == "inferred" and not self.mesh_dims


@dataclass(frozen=True)
class ParameterState:
    spec: TensorSpec
    tensor_state: TensorState | None = None
    layout_shard_dims: tuple[str, ...] = ()
    init_sync: ParameterInitSync = field(default_factory=ParameterInitSync)
    grad_comm: ParameterGradComm = field(default_factory=ParameterGradComm)
    source: str = "inferred"
    explicit_init_sync_none: bool = False
    explicit_grad_comm_none: bool = False

    @classmethod
    def from_spec(cls, spec, *, mesh_dim_names=(), source="inferred"):
        if getattr(spec, "partials", ()):
            raise ValueError("Parameter states must contain only axis layout notation")
        layout_shard_dims = tuple(spec.axes.all_shard_dims())
        annotation = getattr(spec, "annotation", None)
        is_param = bool(getattr(annotation, "is_param", False))
        init_sync = ParameterInitSync.from_annotation(
            getattr(annotation, "init_sync", None),
            layout_shard_dims,
            tuple(mesh_dim_names),
        )
        _validate_parameter_layout(layout_shard_dims, init_sync)
        grad_comm = ParameterGradComm.from_annotation(getattr(annotation, "grad", None), is_param=is_param)
        init_sync_annotation = getattr(annotation, "init_sync", None)
        grad_annotation = getattr(annotation, "grad", None)
        return cls(
            spec=spec,
            tensor_state=_tensor_state_or_none(spec, tuple(mesh_dim_names)),
            layout_shard_dims=layout_shard_dims,
            init_sync=init_sync,
            grad_comm=grad_comm,
            source=source,
            explicit_init_sync_none=init_sync_annotation is not None and init_sync_annotation.mode == "none",
            explicit_grad_comm_none=grad_annotation is not None and grad_annotation.mode == "none",
        )

    @classmethod
    def from_param_spec(cls, spec, *, mesh_dim_names=()):
        init_sync = ParameterInitSync(mode="explicit" if spec.shared else "none", mesh_dims=spec.shared)
        _validate_parameter_layout(tuple(spec.axes.all_shard_dims()), init_sync)
        return cls(
            spec=spec.spec,
            tensor_state=_tensor_state_or_none(spec.spec, tuple(mesh_dim_names)),
            layout_shard_dims=tuple(spec.axes.all_shard_dims()),
            init_sync=init_sync,
            grad_comm=ParameterGradComm.from_reduce_groups(spec.reduce),
            source="ParamSpec",
        )

    @property
    def axes(self):
        return self.spec.axes

    @property
    def shared(self):
        if self.init_sync.mode in {"none", "external"}:
            return ()
        return self.init_sync.mesh_dims

    @property
    def reduce(self):
        if self.grad_comm.mode == "none" or self.grad_comm.backend == "external":
            return ()
        return self.grad_comm.mesh_dims


@dataclass(frozen=True)
class ParamSpec:
    layout: str
    shared: tuple[str, ...] = ()
    reduce: tuple[str, ...] = ()
    spec: TensorSpec = field(init=False, compare=False)
    axes: object = field(init=False)

    def __init__(self, layout, *, shared=(), reduce=()):
        axes = _parse_axes(layout)
        shared = _tuple(shared)
        reduce = _tuple(reduce)
        all_shard_dims = axes.all_shard_dims()
        _validate_parameter_layout(
            all_shard_dims,
            ParameterInitSync(mode="explicit" if shared else "none", mesh_dims=shared),
        )
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "spec", TensorSpec(axes))
        object.__setattr__(self, "shared", shared)
        object.__setattr__(self, "reduce", reduce)

    def __repr__(self):
        args = [repr(self.layout)]
        if self.shared:
            args.append(f"shared={self.shared!r}")
        if self.reduce:
            args.append(f"reduce={self.reduce!r}")
        return f"ParamSpec({', '.join(args)})"


@dataclass(frozen=True)
class ParamShardMetadata:
    global_shape: tuple[int, ...]
    local_slices: tuple[slice, ...]
    local_shape: tuple[int, ...]
    shard_dims: tuple[str, ...]


def _state_from_metadata(metadata):
    if isinstance(metadata, ParameterState):
        return metadata
    if isinstance(metadata, ParamSpec):
        return ParameterState.from_param_spec(metadata)
    raise TypeError("Parameter metadata must be a ParameterState or ParamSpec")


def _require_state(param_or_metadata):
    if isinstance(param_or_metadata, (ParameterState, ParamSpec)):
        return _state_from_metadata(param_or_metadata)
    state = get_parameter_state(param_or_metadata)
    if state is None:
        raise ValueError("Parameter does not have an attached ParameterState or ParamSpec")
    return state


def _native_reduce_groups(state):
    if state is None:
        return ()
    if state.grad_comm.pending_inference:
        if state.grad_comm.backend in {"native", "ddp"}:
            raise ValueError("Parameter gradient communication is still pending inference")
        return ()
    if state.grad_comm.mode == "none" or state.grad_comm.backend != "native":
        return ()
    return state.grad_comm.mesh_dims


class NativeGradReductionHandle:
    def __init__(self):
        self._hooks = []
        self._works = []

    @property
    def pending(self):
        return len(self._works)

    def _add_hook(self, hook):
        self._hooks.append(hook)

    def _add_work(self, work):
        self._works.append(work)

    def wait(self):
        works = self._works
        self._works = []
        for index, work in enumerate(works):
            try:
                work.wait()
            except Exception:
                self._works = works[index:] + self._works
                raise
        return self

    def remove(self):
        self.wait()
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        return self


def _group(mesh, name):
    try:
        return mesh[name].get_group()
    except (KeyError, RuntimeError) as error:
        if "-" in name and not hasattr(mesh, "device_mesh"):
            raise ValueError(
                f"Compound mesh group {name!r} requires es.wrap_mesh(mesh)"
            ) from error
        raise ValueError(f"Mesh does not contain parameter metadata group {name!r}") from error


def sync_param_(param, spec, mesh):
    state = _state_from_metadata(spec)
    for name in state.shared:
        group = _group(mesh, name)
        if dist.get_world_size(group) == 1:
            continue
        dist.broadcast(param.data, src=dist.get_global_rank(group, 0), group=group)
    return param


def reduce_grad_(param, spec, mesh):
    if param.grad is None:
        return param
    state = _state_from_metadata(spec)
    for name in _native_reduce_groups(state):
        param.grad = all_reduce(param.grad, _group(mesh, name))
    return param


def set_param_spec(param, spec):
    setattr(param, PARAM_SPEC_ATTR, spec)
    setattr(param, PARAM_STATE_ATTR, ParameterState.from_param_spec(spec))
    return param


def get_param_spec(param):
    return getattr(param, PARAM_SPEC_ATTR, None)


def set_parameter_state(param, state):
    setattr(param, PARAM_STATE_ATTR, state)
    return param


def get_parameter_state(param):
    state = _parameter_state_from_attached_metadata(param)
    if state is None:
        return None
    if getattr(param, PARAM_STATE_ATTR, None) is None:
        setattr(param, PARAM_STATE_ATTR, state)
    return state


def iter_parameter_states(module):
    for name, param in module.named_parameters():
        state = get_parameter_state(param)
        if state is not None:
            yield name, param, state


def _compatible_init_sync(existing, state):
    if existing == state:
        return True
    if existing.mode in {"none", "external"} or state.mode in {"none", "external"}:
        return False
    return existing.mesh_dims == state.mesh_dims


def _merge_grad_schedule(existing, new):
    if existing.schedule == new.schedule:
        return existing.schedule
    if existing.schedule == "backend_default":
        return new.schedule
    if new.schedule == "backend_default":
        return existing.schedule
    raise ValueError("Parameter is already registered with incompatible metadata")


def _merge_non_none_grad_comm(existing, new):
    if existing.backend != new.backend:
        raise ValueError("Parameter is already registered with incompatible metadata")
    schedule = _merge_grad_schedule(existing, new)
    if existing.pending_inference and new.pending_inference:
        return replace(existing, schedule=schedule)
    if existing.pending_inference or new.pending_inference:
        concrete = new if existing.pending_inference else existing
        pending = existing if existing.pending_inference else new
        if concrete.mode == "explicit":
            return replace(concrete, schedule=schedule)
        return replace(pending, schedule=schedule)
    if existing.mesh_dims != new.mesh_dims:
        raise ValueError("Parameter is already registered with incompatible metadata")
    return replace(existing, schedule=schedule)


def _parameter_state_from_attached_metadata(param):
    state = getattr(param, PARAM_STATE_ATTR, None)
    if state is not None:
        return state
    spec = get_param_spec(param)
    if spec is None:
        return None
    return ParameterState.from_param_spec(spec)


def _explicit_init_sync_none(state):
    if getattr(state, "explicit_init_sync_none", False):
        return True
    annotation = getattr(state.spec, "annotation", None)
    init_sync = getattr(annotation, "init_sync", None)
    return init_sync is not None and init_sync.mode == "none"


def _explicit_grad_comm_none(state):
    if getattr(state, "explicit_grad_comm_none", False):
        return True
    annotation = getattr(state.spec, "annotation", None)
    grad = getattr(annotation, "grad", None)
    return grad is not None and grad.mode == "none"


def _merge_init_sync(existing_state, state):
    existing = existing_state.init_sync
    new = state.init_sync
    if _explicit_init_sync_none(state):
        if existing.mode != "none":
            raise ValueError("Parameter is already registered with incompatible metadata")
        return existing
    if new.mode == "none":
        return existing
    if existing.mode == "none":
        if _explicit_init_sync_none(existing_state):
            raise ValueError("Parameter is already registered with incompatible metadata")
        return new
    if not _compatible_init_sync(existing, new):
        raise ValueError("Parameter is already registered with incompatible metadata")
    return existing


def _merge_grad_comm(existing_state, state):
    existing = existing_state.grad_comm
    new = state.grad_comm
    if _explicit_grad_comm_none(state):
        if existing.mode != "none":
            raise ValueError("Parameter is already registered with incompatible metadata")
        return existing
    if new.mode == "none":
        return existing
    if existing.mode == "none":
        if _explicit_grad_comm_none(existing_state):
            raise ValueError("Parameter is already registered with incompatible metadata")
        return new
    return _merge_non_none_grad_comm(existing, new)


def _merged_source(existing, state):
    if existing.source == "ParamSpec" and state.source != "ParamSpec":
        return "ParamSpec+formula"
    return existing.source


def _merge_tensor_state(existing, state):
    if state.tensor_state is None:
        return existing.tensor_state
    if existing.tensor_state is None:
        return state.tensor_state
    if len(state.tensor_state.replicated_dims) > len(existing.tensor_state.replicated_dims):
        return state.tensor_state
    return existing.tensor_state


def _merge_parameter_state(existing, state):
    if existing is None:
        return state
    if _spec_layout_signature(existing.spec) != _spec_layout_signature(state.spec):
        raise ValueError("Parameter is already registered with a different layout")
    init_sync = _merge_init_sync(existing, state)
    grad_comm = _merge_grad_comm(existing, state)
    tensor_state = _merge_tensor_state(existing, state)
    source = _merged_source(existing, state)
    explicit_init_sync_none = _explicit_init_sync_none(existing) or _explicit_init_sync_none(state)
    explicit_grad_comm_none = _explicit_grad_comm_none(existing) or _explicit_grad_comm_none(state)
    if (
        init_sync == existing.init_sync
        and grad_comm == existing.grad_comm
        and tensor_state == existing.tensor_state
        and source == existing.source
        and explicit_init_sync_none == existing.explicit_init_sync_none
        and explicit_grad_comm_none == existing.explicit_grad_comm_none
    ):
        return existing
    return replace(
        existing,
        spec=state.spec,
        tensor_state=tensor_state,
        init_sync=init_sync,
        grad_comm=grad_comm,
        source=source,
        explicit_init_sync_none=explicit_init_sync_none,
        explicit_grad_comm_none=explicit_grad_comm_none,
    )


def register_parameter_state(param, state):
    existing = _parameter_state_from_attached_metadata(param)
    merged = _merge_parameter_state(existing, state)
    return set_parameter_state(param, merged)


def _infer_parameter_grad_mesh_dims(input_specs, output_spec, operand_index, *, include_axes=True):
    param_spec = input_specs[operand_index]
    param_axis_names = {
        axis.name
        for axis in _flat_axes(param_spec)
        if not isinstance(axis, EllipsisAxis)
    }
    mesh_dims = []

    if include_axes:
        for spec in (*input_specs[:operand_index], *input_specs[operand_index + 1:], output_spec):
            for axis in _flat_axes(spec):
                if isinstance(axis, EllipsisAxis) or axis.name in param_axis_names:
                    continue
                _add_unique_mesh_dim(mesh_dims, axis.shard_dim)

    for partial in getattr(output_spec, "partials", ()):  # incoming output gradients preserve partial obligations
        _add_unique_mesh_dim(mesh_dims, partial)

    return tuple(mesh_dims)


def _with_inferred_parameter_grad_comm(state, input_specs, output_spec, operand_index, *, include_axes=True):
    grad_comm = state.grad_comm
    if not grad_comm.pending_inference:
        return state

    mesh_dims = _infer_parameter_grad_mesh_dims(
        input_specs,
        output_spec,
        operand_index,
        include_axes=include_axes,
    )
    if not mesh_dims:
        grad_comm = ParameterGradComm()
    else:
        grad_comm = replace(grad_comm, mesh_dims=mesh_dims)
    return replace(state, grad_comm=grad_comm)


def parameter_operand_state(
    param,
    input_specs,
    output_spec,
    operand_index,
    *,
    mesh_dim_names=(),
    source="formula",
    infer_grad=True,
):
    if not isinstance(param, torch.nn.Parameter):
        raise TypeError("Annotated parameter operands must be torch.nn.Parameter instances")
    spec = input_specs[operand_index]
    state = ParameterState.from_spec(spec, mesh_dim_names=mesh_dim_names, source=source)
    if not infer_grad:
        return state
    return _with_inferred_parameter_grad_comm(
        state,
        input_specs,
        output_spec,
        operand_index,
        include_axes=infer_grad != "partials",
    )


def register_parameter_operand(
    param,
    input_specs,
    output_spec,
    operand_index,
    *,
    mesh_dim_names=(),
    source="formula",
    infer_grad=True,
):
    state = parameter_operand_state(
        param,
        input_specs,
        output_spec,
        operand_index,
        mesh_dim_names=mesh_dim_names,
        source=source,
        infer_grad=infer_grad,
    )
    return register_parameter_state(param, state)


def iter_param_specs(module):
    for name, param in module.named_parameters():
        spec = get_param_spec(param)
        if spec is not None:
            yield name, param, spec


def _require_spec(param_or_spec):
    if isinstance(param_or_spec, ParamSpec):
        return param_or_spec
    spec = get_param_spec(param_or_spec)
    if spec is None:
        raise ValueError("Parameter does not have an attached ParamSpec")
    return spec


def param_shard_dims(param_or_spec):
    state = _require_state(param_or_spec)
    return state.layout_shard_dims


def _mesh_rank_and_size(mesh, shard_dim):
    if not dist.is_initialized():
        raise RuntimeError("Parameter shard metadata requires an initialized process group")
    device_mesh = getattr(mesh, "device_mesh", mesh)
    if not hasattr(device_mesh, "mesh_dim_names") or not hasattr(device_mesh, "mesh"):
        raise TypeError("mesh must be a PyTorch DeviceMesh or torch_einshard.wrap_mesh(mesh)")
    if shard_dim in device_mesh.mesh_dim_names:
        mesh_dim = device_mesh.mesh_dim_names.index(shard_dim)
        rank = dist.get_rank()
        coord = (device_mesh.mesh == rank).nonzero()[0].tolist()
        return coord[mesh_dim], device_mesh.mesh.shape[mesh_dim]

    if "-" not in shard_dim:
        raise ValueError(f"Mesh does not contain sharded parameter dimension {shard_dim!r}")
    if not hasattr(mesh, "device_mesh"):
        raise ValueError(f"Compound sharded parameter dimension {shard_dim!r} requires es.wrap_mesh(mesh)")

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


def _axis_name(axis):
    if isinstance(axis, EllipsisAxis):
        raise ValueError("Parameter shard metadata does not support ellipsis axes")
    if isinstance(axis, AxisGroup):
        raise NotImplementedError("Parameter shard metadata does not support factored-axis groups")
    return axis.name


def _factor_for(axis, factors):
    if not factors:
        return 1
    name = _axis_name(axis)
    return int(factors.get(name, 1))


def param_local_slices(param_or_spec, global_shape, mesh, factors=None):
    state = _require_state(param_or_spec)
    axes = state.axes
    global_shape = tuple(int(size) for size in global_shape)
    if len(global_shape) != len(axes):
        raise ValueError(
            f"Global shape rank {len(global_shape)} does not match parameter metadata rank {len(axes)}"
        )

    slices = []
    for axis, size in zip(axes, global_shape):
        shard_dim = _axis_shard_dim(axis)
        if not shard_dim:
            slices.append(slice(None))
            continue

        rank, chunks = _mesh_rank_and_size(mesh, shard_dim)
        sections = compute_split_shapes_for_factors(size, chunks, _factor_for(axis, factors))
        start = sum(sections[:rank])
        slices.append(slice(start, start + sections[rank]))
    return tuple(slices)


def param_local_shape(param_or_spec, global_shape, mesh, factors=None):
    slices = param_local_slices(param_or_spec, global_shape, mesh, factors=factors)
    return _shape_from_slices(global_shape, slices)


def _shape_from_slices(global_shape, slices):
    shape = []
    for size, local_slice in zip(global_shape, slices):
        start = 0 if local_slice.start is None else local_slice.start
        stop = size if local_slice.stop is None else local_slice.stop
        shape.append(stop - start)
    return tuple(shape)


def param_shard_metadata(param_or_spec, global_shape, mesh, factors=None):
    state = _require_state(param_or_spec)
    global_shape = tuple(int(size) for size in global_shape)
    local_slices = param_local_slices(state, global_shape, mesh, factors=factors)
    return ParamShardMetadata(
        global_shape=global_shape,
        local_slices=local_slices,
        local_shape=_shape_from_slices(global_shape, local_slices),
        shard_dims=param_shard_dims(state),
    )


def sync_module_params_(module, mesh):
    for param in module.parameters():
        state = get_parameter_state(param)
        if state is not None:
            sync_param_(param, state, mesh)
    return module


def reduce_module_grads_(module, mesh):
    for param in module.parameters():
        state = get_parameter_state(param)
        if state is not None:
            reduce_grad_(param, state, mesh)
    return module


def register_native_grad_reduction_hooks_(module, mesh):
    """Register native per-parameter gradient reductions.

    Every registered parameter must participate in backward in the same order on
    every rank. Use the DDP communication hook or an external backend for models
    with rank-dependent control flow or unused parameters.
    """
    handle = NativeGradReductionHandle()
    hook_specs = []

    for param in module.parameters():
        state = _parameter_state_from_attached_metadata(param)
        if state is None:
            continue

        if state.grad_comm.backend != "native":
            continue

        groups = tuple((name, _group(mesh, name)) for name in _native_reduce_groups(state))
        if not groups:
            continue

        if not param.requires_grad:
            raise ValueError("Cannot register gradient reduction hook for a parameter that does not require gradients")
        hook_specs.append((param, groups))

    try:
        for param, groups in hook_specs:
            def hook(grad, *, groups=groups):
                for _, group in groups:
                    if dist.get_world_size(group) > 1:
                        grad = all_reduce(grad, group)
                return grad

            handle._add_hook(param.register_hook(hook))
    except Exception:
        handle.remove()
        raise

    return handle


def register_grad_reduction_hook_(
    ddp_model,
    mesh,
    ddp_group="dp",
    combined_reduce_group=None,
    combined_reduce=None,
):
    combined_reduce = _tuple(combined_reduce)
    if combined_reduce_group is not None and not combined_reduce:
        raise ValueError("combined_reduce must be provided with combined_reduce_group")

    def bucket_views(bucket):
        offset = 0
        views = []
        buffer = bucket.buffer()
        for param in bucket.parameters():
            view = buffer.narrow(0, offset, param.numel()).view_as(param)
            views.append((param, view, get_parameter_state(param)))
            offset += param.numel()
        return views

    def can_use_combined_reduce(views):
        if combined_reduce_group is None:
            return False
        if not views:
            return False
        expected = set(combined_reduce)
        for _, _, state in views:
            if state is None or set(_native_reduce_groups(state)) != expected:
                return False
        return True

    def reduce_by_parameter_states(buffer, views):
        groups = sorted({
            name
            for _, _, state in views
            for name in _native_reduce_groups(state)
        })
        for name in groups:
            grad_views = [
                view
                for _, view, state in views
                if name in _native_reduce_groups(state)
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

    def reduction_hook(state, bucket):
        buffer = bucket.buffer()
        group = _group(mesh, ddp_group) if ddp_group is not None else None
        group_size = dist.get_world_size(group)
        views = bucket_views(bucket)

        if can_use_combined_reduce(views):
            combined_group = _group(mesh, combined_reduce_group)
            combined_group_size = dist.get_world_size(combined_group)
            if combined_group_size == 1 and group_size == 1:
                future = torch.futures.Future()
                future.set_result(buffer)
            elif combined_group_size > 1:
                future = dist.all_reduce(buffer, group=combined_group, async_op=True).get_future()
            else:
                future = None

            if future is not None:
                def finish_combined(future):
                    if group_size > 1:
                        buffer.div_(group_size)
                    return buffer

                return future.then(finish_combined)

        if group_size == 1:
            future = torch.futures.Future()
            future.set_result(buffer)
        else:
            future = dist.all_reduce(buffer, group=group, async_op=True).get_future()

        def finish(future):
            if group_size > 1:
                buffer.div_(group_size)

            return reduce_by_parameter_states(buffer, views)

        return future.then(finish)

    ddp_model.register_comm_hook(None, reduction_hook)
    return ddp_model
