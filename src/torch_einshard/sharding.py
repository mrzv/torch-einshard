from dataclasses import dataclass


class Axis:
    name: str
    shard_dim: str     # TODO: should this be a list?

    def __init__(self, name: str, shard_dim: str = '') -> None:
        self.name = name
        self.shard_dim = shard_dim

    def local(self) -> bool:
        return not bool(self.shard_dim)

    def __repr__(self):
        if self.local():
            return self.name
        else:
            return f"{self.name} / {self.shard_dim}"

    def __hash__(self):
        return hash(self.name) ^ hash(self.shard_dim)

    def __eq__(self, other: object):
        if not isinstance(other, Axis):
            return NotImplemented

        return self.name == other.name and self.shard_dim == other.shard_dim


class EllipsisAxis:
    name = "..."
    shard_dim = ""

    def local(self) -> bool:
        return True

    def __repr__(self):
        return "..."

    def __hash__(self):
        return hash(EllipsisAxis)

    def __eq__(self, other: object):
        return isinstance(other, EllipsisAxis)


class AxisGroup:
    def __init__(self, axes) -> None:
        self.axes = Axes(axes)

    def local(self) -> bool:
        return self.axes.local()

    def __iter__(self):
        return iter(self.axes)

    def __len__(self):
        return len(self.axes)

    def __repr__(self):
        return f"({' '.join(str(axis) for axis in self.axes)})"

# TODO: eventually add replication, so this won't be just a list
class Axes(list):
    def __repr__(self):
        return " × ".join(str(x) for x in self)

    def local(self):
        for x in self:
            if not x.local():
                return False
        return True

    def all_shard_dims(self):
        dims = []
        for x in self:
            axes = x.axes if isinstance(x, AxisGroup) else [x]
            dims.extend(axis.shard_dim for axis in axes if axis.shard_dim)
        return dims

    def flat(self):
        axes = []
        for x in self:
            if isinstance(x, AxisGroup):
                axes.extend(x.axes)
            else:
                axes.append(x)
        return Axes(axes)


@dataclass(frozen=True)
class GradAnnotation:
    mode: str = "inferred"
    mesh_dims: tuple[str, ...] = ()
    backend: str = "native"
    schedule: str = "backend_default"

    @classmethod
    def from_value(cls, value):
        if ":" in value:
            mesh_dim, suffix = value.rsplit(":", 1)
            if not mesh_dim or not suffix:
                raise ValueError("Invalid grad annotation")
            if mesh_dim in {"async", "ddp", "external", "none"}:
                raise ValueError("Grad annotation suffix requires an explicit mesh group")
            if suffix == "async":
                return cls(mode="explicit", mesh_dims=(mesh_dim,), backend="native", schedule="async")
            if suffix == "ddp":
                return cls(mode="explicit", mesh_dims=(mesh_dim,), backend="ddp")
            if suffix == "external":
                return cls(mode="explicit", mesh_dims=(mesh_dim,), backend="external")
            raise ValueError(f"Unknown grad annotation suffix {suffix!r}")

        if value == "async":
            return cls(schedule="async")
        if value == "ddp":
            return cls(backend="ddp")
        if value == "external":
            return cls(backend="external")
        if value == "none":
            return cls(mode="none", backend="none")
        return cls(mode="explicit", mesh_dims=(value,))

    def __repr__(self):
        if self.mode == "none":
            return "none"
        if self.mode == "inferred" and self.backend == "native" and self.schedule == "async":
            return "async"
        if self.mode == "inferred" and self.backend in {"ddp", "external"}:
            return self.backend
        if self.mode == "explicit" and self.mesh_dims:
            value = self.mesh_dims[0]
            if self.backend == "native" and self.schedule == "async":
                return f"{value}:async"
            if self.backend in {"ddp", "external"}:
                return f"{value}:{self.backend}"
            return value
        return "inferred"


@dataclass(frozen=True)
class InitSyncAnnotation:
    mode: str = "inferred"
    mesh_dims: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value):
        if ":" in value:
            raise ValueError("init_sync annotations do not accept suffixes")
        if value == "none":
            return cls(mode="none")
        if value == "external":
            return cls(mode="external")
        return cls(mode="explicit", mesh_dims=(value,))

    def __repr__(self):
        if self.mode in {"none", "external"}:
            return self.mode
        if self.mode == "explicit" and self.mesh_dims:
            return self.mesh_dims[0]
        return "inferred"


@dataclass(frozen=True)
class TensorAnnotation:
    is_param: bool = False
    grad: GradAnnotation | None = None
    init_sync: InitSyncAnnotation | None = None

    @classmethod
    def from_items(cls, items):
        is_param = False
        grad = None
        init_sync = None
        seen = set()
        for key, value in items:
            if key in seen:
                raise ValueError(f"Duplicate tensor annotation {key!r}")
            seen.add(key)
            if key == "param":
                is_param = True
            elif key == "grad":
                grad = GradAnnotation.from_value(value)
            elif key == "init_sync":
                init_sync = InitSyncAnnotation.from_value(value)
            else:
                raise ValueError(f"Unknown tensor annotation {key!r}")
        return cls(is_param=is_param, grad=grad, init_sync=init_sync)

    def __bool__(self):
        return self.is_param or self.grad is not None or self.init_sync is not None

    def __repr__(self):
        items = []
        if self.is_param:
            items.append("param")
        if self.grad is not None:
            items.append(f"grad={self.grad!r}")
        if self.init_sync is not None:
            items.append(f"init_sync={self.init_sync!r}")
        return f"[{', '.join(items)}]"


class TensorSpec:
    def __init__(self, axes: Axes, partials = None, annotation = None) -> None:
        if sum(isinstance(axis, EllipsisAxis) for axis in axes) > 1:
            raise ValueError("Tensor specs may contain at most one ellipsis")
        self.axes = axes
        self.partials = tuple(partials or [])
        self.annotation = annotation or TensorAnnotation()

    def local(self):
        return self.axes.local() and not self.partials

    def __iter__(self):
        return iter(self.axes)

    def __len__(self):
        return len(self.axes)

    def __getitem__(self, key):
        return self.axes[key]

    def __repr__(self):
        result = repr(self.axes)
        if self.partials:
            partials = self.partials[0] if len(self.partials) == 1 else f"({','.join(self.partials)})"
            result += f" // {partials}"
        if self.annotation:
            result += f" {self.annotation!r}"
        return result
