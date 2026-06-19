from dataclasses import dataclass

from .sharding import Axis, Axes, EllipsisAxis, TensorSpec


def axes_of(spec):
    return spec.axes if hasattr(spec, "axes") else spec


def flat_axes_of(spec):
    axes = axes_of(spec)
    return axes.flat() if hasattr(axes, "flat") else axes


def partials_of(spec):
    return tuple(getattr(spec, "partials", ()))


def require_expanded_axes(spec):
    axes = flat_axes_of(spec)
    if any(isinstance(axis, EllipsisAxis) for axis in axes):
        raise ValueError("Symbolic TensorState requires ellipsis expansion")
    return Axes(axes)


@dataclass(frozen=True)
class StateAxis:
    name: str
    shard_dim: str = ""

    def local(self):
        return not bool(self.shard_dim)


@dataclass(frozen=True)
class TensorState:
    axes: tuple
    placements: tuple
    partials: tuple

    @classmethod
    def from_spec(cls, spec):
        axes = tuple(StateAxis(axis.name, axis.shard_dim) for axis in require_expanded_axes(spec))
        names = [axis.name for axis in axes]
        if len(set(names)) != len(names):
            raise ValueError("Symbolic TensorState requires unique axis names")
        return cls(
            axes=axes,
            placements=tuple((axis.name, axis.shard_dim or None) for axis in axes),
            partials=partials_of(spec),
        )

    def placement(self, name):
        for axis_name, shard_dim in self.placements:
            if axis_name == name:
                return shard_dim
        raise KeyError(name)

    def placement_dict(self):
        return dict(self.placements)

    def axis(self, name):
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise KeyError(name)


@dataclass(frozen=True)
class UnaryAxisDelta:
    name: str
    source: str | None
    target: str | None


@dataclass(frozen=True)
class UnaryClassification:
    input_state: TensorState
    output_state: TensorState
    placement_deltas: tuple
    removed_partials: tuple
    added_partials: tuple


@dataclass(frozen=True)
class BinaryClassification:
    input0_state: TensorState
    input1_state: TensorState
    output_state: TensorState
    free_axes: tuple
    shared_output_axes: tuple
    contracted_axes: tuple
    output_only_axes: tuple


@dataclass(frozen=True)
class PlanStep:
    name: str
    args: tuple = ()


class ExecutionPlan:
    def __init__(self):
        self.steps = []

    def add(self, name, *args):
        self.steps.append(PlanStep(name, tuple(args)))

    def names(self):
        return [step.name for step in self.steps]

    def snapshot(self):
        return tuple(self.steps)


_last_plan = ()


def set_last_plan(plan):
    global _last_plan
    _last_plan = tuple(plan.steps if isinstance(plan, ExecutionPlan) else plan)


def last_plan():
    return _last_plan


def classify_unary(input_spec, output_spec):
    input_state = TensorState.from_spec(input_spec)
    output_state = TensorState.from_spec(output_spec)
    input_names = {axis.name for axis in input_state.axes}
    output_names = {axis.name for axis in output_state.axes}
    if input_names != output_names:
        raise ValueError("Input and output axes must match")

    deltas = []
    for axis in output_state.axes:
        source = input_state.placement(axis.name)
        target = output_state.placement(axis.name)
        if source != target:
            deltas.append(UnaryAxisDelta(axis.name, source, target))

    return UnaryClassification(
        input_state=input_state,
        output_state=output_state,
        placement_deltas=tuple(deltas),
        removed_partials=tuple(partial for partial in input_state.partials if partial not in output_state.partials),
        added_partials=tuple(partial for partial in output_state.partials if partial not in input_state.partials),
    )


def classify_binary(input0_spec, input1_spec, output_spec):
    input0_state = TensorState.from_spec(input0_spec)
    input1_state = TensorState.from_spec(input1_spec)
    output_state = TensorState.from_spec(output_spec)
    input0_names = {axis.name for axis in input0_state.axes}
    input1_names = {axis.name for axis in input1_state.axes}
    output_names = {axis.name for axis in output_state.axes}
    input_names = input0_names | input1_names

    return BinaryClassification(
        input0_state=input0_state,
        input1_state=input1_state,
        output_state=output_state,
        free_axes=tuple(axis.name for axis in output_state.axes if axis.name in input_names and axis.name not in input0_names & input1_names),
        shared_output_axes=tuple(axis.name for axis in output_state.axes if axis.name in input0_names and axis.name in input1_names),
        contracted_axes=tuple(axis.name for axis in input0_state.axes if axis.name in input1_names and axis.name not in output_names),
        output_only_axes=tuple(axis.name for axis in output_state.axes if axis.name not in input_names),
    )


def tensor_spec(axes, partials=()):
    return TensorSpec(Axes(axes), partials)


def local_axis(name):
    return Axis(name)


def sharded_axis(name, shard_dim):
    return Axis(name, shard_dim)
