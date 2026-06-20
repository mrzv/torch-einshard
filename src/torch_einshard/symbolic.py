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
class UnaryTransitionPlan:
    classification: UnaryClassification
    steps: tuple

    def names(self):
        return [step.name for step in self.steps]


@dataclass(frozen=True)
class BinaryPostRepartition:
    shard_dim: str
    source_axis: str
    dest_axis: str


@dataclass(frozen=True)
class PlanCandidate:
    name: str
    args: tuple = ()
    fallback_steps: tuple = ()
    status: str = "considered"
    reason: str = ""

    def with_result(self, status, reason=""):
        return PlanCandidate(self.name, self.args, self.fallback_steps, status, reason)


@dataclass(frozen=True)
class BinaryTransitionPlan:
    classification: BinaryClassification
    input0_steps: tuple
    input1_steps: tuple
    output_steps: tuple
    reduction_dims: tuple
    scatter_output_by_dim: tuple
    candidates: tuple = ()
    post_repartition: BinaryPostRepartition | None = None

    def names(self):
        return [step.name for step in self.steps()]

    def steps(self):
        return (*self.input0_steps, *self.input1_steps, *self.output_steps)


@dataclass(frozen=True)
class PlanStep:
    name: str
    args: tuple = ()


class ExecutionPlan:
    def __init__(self):
        self.steps = []
        self.candidates = []

    def add(self, name, *args):
        self.steps.append(PlanStep(name, tuple(args)))

    def execute(self, name, fn, *fn_args, step_args=(), **fn_kwargs):
        self.add(name, *step_args)
        return fn(*fn_args, **fn_kwargs)

    def consider(self, candidate, *args, status="considered", reason="", fallback_steps=()):
        if isinstance(candidate, PlanCandidate):
            self.candidates.append(candidate.with_result(status, reason))
        else:
            self.candidates.append(PlanCandidate(candidate, tuple(args), tuple(fallback_steps), status, reason))

    def names(self):
        return [step.name for step in self.steps]

    def snapshot(self):
        return tuple(self.steps)


_last_plan = ()
_last_candidates = ()


def set_last_plan(plan):
    global _last_plan, _last_candidates
    _last_plan = tuple(plan.steps if isinstance(plan, ExecutionPlan) else plan)
    _last_candidates = tuple(plan.candidates if isinstance(plan, ExecutionPlan) else ())


def last_plan():
    return _last_plan


def last_candidates():
    return _last_candidates


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


def build_unary_transition_plan(input_spec, output_spec):
    classification = classify_unary(input_spec, output_spec)
    input_state = classification.input_state
    output_state = classification.output_state
    current_placements = input_state.placement_dict()
    current_partials = list(input_state.partials)
    steps = []

    for partial in classification.removed_partials:
        scatter_axis = None
        for axis in output_state.axes:
            if input_state.placement(axis.name) is None and output_state.placement(axis.name) == partial:
                scatter_axis = axis
                break

        if scatter_axis is None:
            steps.append(PlanStep("allreduce_forward_identity_backward", (partial,)))
        else:
            steps.append(PlanStep("reducescatter_forward_allgather_backward", (scatter_axis.name, partial)))
            current_placements[scatter_axis.name] = partial
        current_partials.remove(partial)

    for axis in output_state.axes:
        source = current_placements[axis.name]
        target = output_state.placement(axis.name)
        if source == target:
            continue
        if source is None:
            steps.append(PlanStep("split_forward_allgather_backward", (axis.name, target)))
        elif target is None:
            if source in output_state.partials:
                steps.append(PlanStep("allgather_forward_reducescatter_backward", (axis.name, source)))
                current_partials.append(source)
            else:
                steps.append(PlanStep("allgather_forward_split_backward", (axis.name, source)))
        else:
            steps.append(PlanStep("allgather_forward_split_backward", (axis.name, source)))
            steps.append(PlanStep("split_forward_allgather_backward", (axis.name, target)))
        current_placements[axis.name] = target

    for partial in classification.added_partials:
        if partial in current_partials:
            continue
        steps.append(PlanStep("identity_forward_allreduce_backward", (partial,)))
        current_partials.append(partial)

    if tuple(axis.name for axis in input_state.axes) != tuple(axis.name for axis in output_state.axes):
        steps.append(PlanStep("permute", (tuple(axis.name for axis in output_state.axes),)))

    return UnaryTransitionPlan(classification, tuple(steps))


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


def _target_axis_for_contracted(name, input0_state, input1_state):
    axis0 = input0_state.axis(name)
    axis1 = input1_state.axis(name)
    if axis0 == axis1:
        return axis0
    if axis0.local() and not axis1.local():
        return axis1
    if axis1.local() and not axis0.local():
        return axis0
    if axis0.shard_dim and axis1.shard_dim:
        return axis0
    raise NotImplementedError("Unsupported contracted-axis sharding mismatch")


def _normalize_axis_steps(axis, target_axis, active_output_dims):
    if axis.local():
        return (PlanStep("split_forward_allgather_backward", (axis.name, target_axis.shard_dim)),)
    if target_axis.local():
        if axis.shard_dim in active_output_dims:
            return (PlanStep("allgather_forward_reducescatter_backward", (axis.name, axis.shard_dim)),)
        return (PlanStep("allgather_forward_split_backward", (axis.name, axis.shard_dim)),)

    steps = []
    if axis.shard_dim in active_output_dims:
        steps.append(PlanStep("allgather_forward_reducescatter_backward", (axis.name, axis.shard_dim)))
    else:
        steps.append(PlanStep("allgather_forward_split_backward", (axis.name, axis.shard_dim)))
    steps.append(PlanStep("split_forward_allgather_backward", (axis.name, target_axis.shard_dim)))
    return tuple(steps)


def build_binary_transition_plan(input0_spec, input1_spec, output_spec):
    classification = classify_binary(input0_spec, input1_spec, output_spec)
    input0_state = classification.input0_state
    input1_state = classification.input1_state
    output_state = classification.output_state
    input0_by_name = {axis.name: axis for axis in input0_state.axes}
    input1_by_name = {axis.name: axis for axis in input1_state.axes}
    output_by_name = {axis.name: axis for axis in output_state.axes}
    output_partials = set(output_state.partials)

    contracted_target_by_name = {
        name: _target_axis_for_contracted(name, input0_state, input1_state)
        for name in classification.contracted_axes
    }
    reduction_dims = []
    for name in classification.contracted_axes:
        axis = contracted_target_by_name[name]
        if axis.local() or axis.shard_dim in reduction_dims:
            continue
        reduction_dims.append(axis.shard_dim)

    scatter_output_by_dim = {}
    for axis in output_state.axes:
        matching_input_axes = [
            input_axis for input_axis in (input0_by_name.get(axis.name), input1_by_name.get(axis.name))
            if input_axis is not None
        ]
        if axis.shard_dim in reduction_dims and any(input_axis.local() or input_axis == axis for input_axis in matching_input_axes):
            scatter_output_by_dim[axis.shard_dim] = axis.name

    gather_candidates = []
    split_candidates = []
    for output_axis in output_state.axes:
        matching_input_axes = [
            input_axis for input_axis in (input0_by_name.get(output_axis.name), input1_by_name.get(output_axis.name))
            if input_axis is not None
        ]
        if len(matching_input_axes) != 1:
            continue
        input_axis = matching_input_axes[0]
        if input_axis == output_axis:
            continue
        if input_axis.shard_dim and output_axis.local():
            gather_candidates.append((input_axis, output_axis))
        elif input_axis.local() and output_axis.shard_dim:
            split_candidates.append((input_axis, output_axis))

    candidates = []
    post_repartition = None
    if len(gather_candidates) == 1 and len(split_candidates) == 1:
        source_input_axis, _ = gather_candidates[0]
        _, dest_output_axis = split_candidates[0]
        shard_dim = source_input_axis.shard_dim
        if shard_dim == dest_output_axis.shard_dim and shard_dim not in reduction_dims and shard_dim not in output_partials:
            post_repartition = BinaryPostRepartition(
                shard_dim=shard_dim,
                source_axis=source_input_axis.name,
                dest_axis=dest_output_axis.name,
            )
            candidates.append(
                PlanCandidate(
                    "alltoall_repartition",
                    (source_input_axis.name, dest_output_axis.name, shard_dim),
                    (
                        PlanStep("allgather_forward_reducescatter_backward", (source_input_axis.name, shard_dim)),
                        PlanStep("split_forward_allgather_backward", (dest_output_axis.name, shard_dim)),
                    ),
                )
            )

    active_output_dims = output_partials | {axis.shard_dim for axis in output_state.axes if axis.shard_dim}

    def normalize_input_steps(state):
        steps = []
        normalized_scatter_axes = set()
        owner_swapped_axes = set()
        owner_swap_changes = []
        for axis in state.axes:
            target_axis = contracted_target_by_name.get(axis.name)
            if target_axis is None or axis == target_axis:
                continue
            if axis.local() or target_axis.local():
                continue
            owner_swap_changes.append((axis, target_axis))

        if len(owner_swap_changes) >= 2:
            source_shard_dims = tuple(axis.shard_dim for axis, _ in owner_swap_changes)
            dest_shard_dims = tuple(target_axis.shard_dim for _, target_axis in owner_swap_changes)
            involved_shard_dims = set(source_shard_dims) | set(dest_shard_dims)
            changed_names = {axis.name for axis, _ in owner_swap_changes}
            can_owner_swap = (
                len(set(source_shard_dims)) == len(source_shard_dims)
                and len(set(dest_shard_dims)) == len(dest_shard_dims)
                and set(source_shard_dims) == set(dest_shard_dims)
            )
            can_owner_swap = can_owner_swap and all(
                axis.name in changed_names or axis.shard_dim not in involved_shard_dims
                for axis in state.axes
            )
            if not can_owner_swap:
                raise NotImplementedError(
                    "Multiple contracted-axis shard-dimension changes require an owner-swap-compatible permutation"
                )
            owner_swap_step = PlanStep("owner_swap", (source_shard_dims, dest_shard_dims))
            steps.append(owner_swap_step)
            candidates.append(PlanCandidate("owner_swap", owner_swap_step.args))
            owner_swapped_axes.update(changed_names)

        for axis in state.axes:
            if axis.name in contracted_target_by_name or axis.name not in output_by_name:
                continue
            output_axis = output_by_name[axis.name]
            if scatter_output_by_dim.get(output_axis.shard_dim) != output_axis.name or axis != output_axis:
                continue
            steps.append(PlanStep("allgather_forward_split_backward", (axis.name, axis.shard_dim)))
            normalized_scatter_axes.add(axis.name)

        for axis in state.axes:
            if axis.name in normalized_scatter_axes:
                continue
            if axis.name in contracted_target_by_name:
                if axis.name in owner_swapped_axes:
                    continue
                target_axis = contracted_target_by_name[axis.name]
                if axis != target_axis:
                    steps.extend(_normalize_axis_steps(axis, target_axis, active_output_dims))
                continue

            if axis.name not in output_by_name:
                continue
            output_axis = output_by_name[axis.name]
            if axis == output_axis:
                continue
            if axis.local() and output_axis.shard_dim in reduction_dims:
                continue
            steps.extend(_normalize_axis_steps(axis, output_axis, active_output_dims))
        return tuple(steps)

    output_steps = [PlanStep("rank_local_einsum")]
    for shard_dim in reduction_dims:
        if shard_dim in output_partials:
            scatter_axis_name = scatter_output_by_dim.get(shard_dim)
            if scatter_axis_name is not None:
                output_steps.append(PlanStep("split_forward_allgather_backward", (scatter_axis_name, shard_dim)))
            continue

        scatter_axis_name = scatter_output_by_dim.get(shard_dim)
        if scatter_axis_name is None:
            output_steps.append(PlanStep("allreduce_forward_identity_backward", (shard_dim,)))
        else:
            output_steps.append(PlanStep("reducescatter_forward_allgather_backward", (scatter_axis_name, shard_dim)))

    return BinaryTransitionPlan(
        classification=classification,
        input0_steps=normalize_input_steps(input0_state),
        input1_steps=normalize_input_steps(input1_state),
        output_steps=tuple(output_steps),
        reduction_dims=tuple(reduction_dims),
        scatter_output_by_dim=tuple(scatter_output_by_dim.items()),
        candidates=tuple(candidates),
        post_repartition=post_repartition,
    )


def tensor_spec(axes, partials=()):
    return TensorSpec(Axes(axes), partials)


def local_axis(name):
    return Axis(name)


def sharded_axis(name, shard_dim):
    return Axis(name, shard_dim)
