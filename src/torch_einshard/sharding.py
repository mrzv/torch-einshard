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

class TensorSpec:
    def __init__(self, axes: Axes, partials = None) -> None:
        if sum(isinstance(axis, EllipsisAxis) for axis in axes) > 1:
            raise ValueError("Tensor specs may contain at most one ellipsis")
        self.axes = axes
        self.partials = tuple(partials or [])

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
        return result
