from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .sharding import Axis, Axes, EllipsisAxis, TensorSpec


def axes_of(spec):
    return spec.axes if hasattr(spec, "axes") else spec


def flat_axes_of(spec):
    axes = axes_of(spec)
    return axes.flat() if hasattr(axes, "flat") else axes


def partials_of(spec):
    return tuple(getattr(spec, "partials", ()))


def mesh_dim_components(name):
    return {name, *name.split("-")}


def mesh_dims_components(names):
    result = set()
    for name in names:
        result.update(mesh_dim_components(name))
    return result


def require_expanded_axes(spec):
    axes = flat_axes_of(spec)
    if any(isinstance(axis, EllipsisAxis) for axis in axes):
        raise ValueError("Symbolic TensorState requires ellipsis expansion")
    return Axes(axes)


_transition_plan_cache = {}
_context_policy = ContextVar("torch_einshard_plan_policy", default=None)
_default_policy = None


def _spec_cache_key(spec):
    axes = tuple((axis.name, axis.shard_dim) for axis in require_expanded_axes(spec))
    return axes, partials_of(spec)


def clear_plan_cache():
    _transition_plan_cache.clear()


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
    replicated_dims: tuple = ()

    @classmethod
    def from_spec(cls, spec, mesh_dim_names=()):
        axes = tuple(StateAxis(axis.name, axis.shard_dim) for axis in require_expanded_axes(spec))
        names = [axis.name for axis in axes]
        if len(set(names)) != len(names):
            raise ValueError("Symbolic TensorState requires unique axis names")
        used_dims = mesh_dims_components(
            [axis.shard_dim for axis in axes if axis.shard_dim] + list(partials_of(spec))
        )
        return cls(
            axes=axes,
            placements=tuple((axis.name, axis.shard_dim or None) for axis in axes),
            partials=partials_of(spec),
            replicated_dims=tuple(
                dim for dim in mesh_dim_names
                if not mesh_dim_components(dim).intersection(used_dims)
            ),
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
class PlanCost:
    score: int
    collectives: int = 0
    materializations: int = 0
    peak_factor: int = 1
    forward_bytes: int = 0
    backward_bytes: int = 0
    peak_elements: int = 0
    materialized_elements: int = 0
    requires_shapes: bool = False
    invalid_reason: str = ""

    @property
    def total_bytes(self):
        return self.forward_bytes + self.backward_bytes

    def with_score(self, score):
        return PlanCost(
            score,
            self.collectives,
            self.materializations,
            self.peak_factor,
            self.forward_bytes,
            self.backward_bytes,
            self.peak_elements,
            self.materialized_elements,
            self.requires_shapes,
            self.invalid_reason,
        )


@dataclass(frozen=True)
class PlanPolicy:
    mode: str = "training"
    forward_byte_weight: int = 1
    backward_byte_weight: int = 1
    peak_element_weight: int = 1
    materialized_element_weight: int = 1
    collective_weight: int = 100
    materialization_weight: int = 200
    base_step_weight: int = 1

    @staticmethod
    def from_mode(mode):
        if isinstance(mode, PlanPolicy):
            return mode
        if mode is None:
            return PlanPolicy()
        if mode == "training":
            return PlanPolicy(mode="training")
        if mode == "inference":
            return PlanPolicy(mode="inference", backward_byte_weight=0)
        if mode == "memory":
            return PlanPolicy(
                mode="memory",
                forward_byte_weight=1,
                backward_byte_weight=1,
                peak_element_weight=64,
                materialized_element_weight=64,
                collective_weight=50,
                materialization_weight=400,
            )
        if mode == "communication":
            return PlanPolicy(
                mode="communication",
                forward_byte_weight=8,
                backward_byte_weight=8,
                peak_element_weight=1,
                materialized_element_weight=1,
                collective_weight=50,
                materialization_weight=100,
            )
        if mode == "latency":
            return PlanPolicy(
                mode="latency",
                forward_byte_weight=1,
                backward_byte_weight=1,
                peak_element_weight=1,
                materialized_element_weight=1,
                collective_weight=1000,
                materialization_weight=100,
            )
        raise ValueError(f"Unknown optimization policy {mode!r}")

    def score(self, cost):
        return (
            cost.score * self.base_step_weight
            + cost.collectives * self.collective_weight
            + cost.materializations * self.materialization_weight
            + (cost.forward_bytes * self.forward_byte_weight + cost.backward_bytes * self.backward_byte_weight) // 1024
            + (cost.peak_elements * self.peak_element_weight) // 1024
            + (cost.materialized_elements * self.materialized_element_weight) // 1024
        )


def _coerce_policy(value):
    return value if isinstance(value, PlanPolicy) else PlanPolicy.from_mode(value)


def resolve_plan_policy(optimize=None, policy=None):
    if optimize is not None and policy is not None:
        raise ValueError("Pass either optimize or policy, not both")
    if policy is not None:
        return _coerce_policy(policy)
    if optimize is not None:
        return _coerce_policy(optimize)
    context_policy = _context_policy.get()
    if context_policy is not None:
        return context_policy
    if _default_policy is not None:
        return _default_policy
    return PlanPolicy.from_mode("training")


@contextmanager
def optimize(policy):
    token = _context_policy.set(_coerce_policy(policy))
    try:
        yield
    finally:
        _context_policy.reset(token)


def set_default_policy(policy):
    global _default_policy
    _default_policy = None if policy is None else _coerce_policy(policy)


def get_default_policy():
    return _default_policy if _default_policy is not None else PlanPolicy.from_mode("training")


def get_optimization_policy():
    return resolve_plan_policy()


@dataclass(frozen=True)
class TensorRuntimeInfo:
    axes: tuple
    shape: tuple
    dtype_size: int

    def dim(self, axis_name):
        for index, axis in enumerate(self.axes):
            if axis.name == axis_name:
                return index
        return None

    def num_elements(self):
        elements = 1
        for size in self.shape:
            elements *= size
        return elements

    def with_axis_size(self, axis_name, size):
        dim = self.dim(axis_name)
        if dim is None:
            return self
        shape = list(self.shape)
        shape[dim] = size
        return TensorRuntimeInfo(self.axes, tuple(shape), self.dtype_size)

    def with_axis(self, axis_name, replacement):
        axes = tuple(replacement if axis.name == axis_name else axis for axis in self.axes)
        return TensorRuntimeInfo(axes, self.shape, self.dtype_size)


@dataclass(frozen=True)
class PlanAlternative:
    name: str
    input0_steps: tuple = ()
    input1_steps: tuple = ()
    output_steps: tuple = ()
    candidates: tuple = ()
    cost: PlanCost = PlanCost(0)
    status: str = "ranked"
    reason: str = ""

    def steps(self):
        return (*self.input0_steps, *self.input1_steps, *self.output_steps)

    def with_result(self, status, reason=""):
        return PlanAlternative(
            self.name,
            self.input0_steps,
            self.input1_steps,
            self.output_steps,
            self.candidates,
            self.cost,
            status,
            reason,
        )

    def with_cost(self, cost):
        return PlanAlternative(
            self.name,
            self.input0_steps,
            self.input1_steps,
            self.output_steps,
            self.candidates,
            cost,
            self.status,
            self.reason,
        )


@dataclass(frozen=True)
class BinaryTransitionPlan:
    classification: BinaryClassification
    input0_steps: tuple
    input1_steps: tuple
    output_steps: tuple
    reduction_dims: tuple
    scatter_output_by_dim: tuple
    candidates: tuple = ()
    alternatives: tuple = ()
    top_ranked_alternative: str = ""
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
        self.alternatives = []

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

    def rank(self, alternatives, selected=None, reason=""):
        self.alternatives = [
            alternative.with_result("selected", reason) if alternative.name == selected else alternative
            for alternative in alternatives
        ]

    def names(self):
        return [step.name for step in self.steps]

    def snapshot(self):
        return tuple(self.steps)


_last_plan = ()
_last_candidates = ()
_last_alternatives = ()


def set_last_plan(plan):
    global _last_plan, _last_candidates, _last_alternatives
    _last_plan = tuple(plan.steps if isinstance(plan, ExecutionPlan) else plan)
    _last_candidates = tuple(plan.candidates if isinstance(plan, ExecutionPlan) else ())
    _last_alternatives = tuple(plan.alternatives if isinstance(plan, ExecutionPlan) else ())


def last_plan():
    return _last_plan


def last_candidates():
    return _last_candidates


def last_alternatives():
    return _last_alternatives


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
    cache_key = ("unary", _spec_cache_key(input_spec), _spec_cache_key(output_spec))
    cached = _transition_plan_cache.get(cache_key)
    if cached is not None:
        return cached

    classification = classify_unary(input_spec, output_spec)
    input_state = classification.input_state
    output_state = classification.output_state
    current_placements = input_state.placement_dict()
    current_partials = list(input_state.partials)
    steps = []

    def owner_swap_changes():
        if output_state.partials or tuple(axis.name for axis in input_state.axes) != tuple(axis.name for axis in output_state.axes):
            return ()
        changed = []
        for output_axis in output_state.axes:
            input_axis = input_state.axis(output_axis.name)
            if input_axis.shard_dim and output_axis.shard_dim and input_axis.shard_dim != output_axis.shard_dim:
                changed.append((input_axis, output_axis))
        if not changed:
            return ()
        source_shard_dims = tuple(input_axis.shard_dim for input_axis, _ in changed)
        dest_shard_dims = tuple(output_axis.shard_dim for _, output_axis in changed)
        if len(set(source_shard_dims)) != len(source_shard_dims) or len(set(dest_shard_dims)) != len(dest_shard_dims):
            return ()
        if len(changed) > 1 and set(source_shard_dims) != set(dest_shard_dims):
            return ()
        involved_dims = set(source_shard_dims) | set(dest_shard_dims)
        changed_names = {input_axis.name for input_axis, _ in changed}
        for input_axis, output_axis in zip(input_state.axes, output_state.axes):
            if input_axis.name in changed_names:
                continue
            if input_axis.shard_dim in involved_dims or output_axis.shard_dim in involved_dims:
                return ()
        return tuple(changed)

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

    swapped_axes = owner_swap_changes()
    swapped_names = {input_axis.name for input_axis, _ in swapped_axes}
    if swapped_axes:
        steps.append(PlanStep(
            "owner_swap",
            (
                tuple(input_axis.shard_dim for input_axis, _ in swapped_axes),
                tuple(output_axis.shard_dim for _, output_axis in swapped_axes),
            ),
        ))
        for input_axis, output_axis in swapped_axes:
            current_placements[input_axis.name] = output_axis.shard_dim

    for axis in output_state.axes:
        if axis.name in swapped_names:
            continue
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

    plan = UnaryTransitionPlan(classification, tuple(steps))
    _transition_plan_cache[cache_key] = plan
    return plan


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


def _step_cost(step):
    if step.name == "rank_local_einsum":
        return PlanCost(0)
    if step.name == "reducescatter_forward_allgather_backward":
        return PlanCost(5, collectives=1)
    if step.name == "alltoall_repartition":
        return PlanCost(4, collectives=1)
    if step.name == "owner_swap":
        return PlanCost(6, collectives=1)
    if step.name == "broadcast_forward_allreduce_backward":
        return PlanCost(7, collectives=1)
    if step.name == "split_forward_allgather_backward":
        return PlanCost(3, collectives=1)
    if step.name == "allreduce_forward_identity_backward":
        return PlanCost(8, collectives=1)
    if step.name == "identity_forward_allreduce_backward":
        return PlanCost(8, collectives=1)
    if step.name == "allgather_forward_split_backward":
        return PlanCost(10, collectives=1, materializations=1, peak_factor=2)
    if step.name == "allgather_forward_reducescatter_backward":
        return PlanCost(10, collectives=1, materializations=1, peak_factor=2)
    return PlanCost(20, collectives=1)


def _axis_size_from_split_shapes(split_shapes, shard_dim, axis_name, direction):
    axis_shapes = None
    if split_shapes is not None:
        axis_shapes = split_shapes.get((shard_dim, axis_name))
    if axis_shapes is None:
        return None
    if direction == "split":
        return max(axis_shapes)
    return sum(axis_shapes)


def _mesh_size(mesh_sizes, shard_dim):
    if mesh_sizes is None:
        return 1
    return mesh_sizes.get(shard_dim, 1)


def _rescale_axis(info, axis_name, shard_dim, mesh_sizes, split_shapes, direction):
    dim = info.dim(axis_name)
    if dim is None:
        return info

    shape_size = _axis_size_from_split_shapes(split_shapes, shard_dim, axis_name, direction)
    if shape_size is None:
        if direction == "gather":
            shape_size = info.shape[dim] * _mesh_size(mesh_sizes, shard_dim)
        elif direction == "split":
            mesh_size = max(_mesh_size(mesh_sizes, shard_dim), 1)
            shape_size = max(1, (info.shape[dim] + mesh_size - 1) // mesh_size)
        else:
            shape_size = info.shape[dim]
    return info.with_axis_size(axis_name, shape_size)


def _dynamic_step_cost(step, info, mesh_sizes, split_shapes):
    static = _step_cost(step)
    if info is None:
        return static, info

    before_elements = info.num_elements()
    after_info = info
    forward_elements = before_elements
    backward_elements = before_elements
    materialized_elements = 0
    peak_elements = before_elements
    requires_shapes = False

    if step.name == "rank_local_einsum":
        return PlanCost(0, peak_elements=before_elements), info

    if step.name in {"split_forward_allgather_backward", "reducescatter_forward_allgather_backward"}:
        axis_name, shard_dim = step.args
        axis_shapes = split_shapes.get((shard_dim, axis_name)) if split_shapes is not None else None
        after_info = _rescale_axis(info, axis_name, shard_dim, mesh_sizes, split_shapes, "split")
        forward_elements = before_elements
        backward_elements = max(before_elements, after_info.num_elements())
        peak_elements = max(before_elements, after_info.num_elements())
        if step.name == "reducescatter_forward_allgather_backward" and axis_shapes is not None and len(set(axis_shapes)) != 1:
            full_info = _rescale_axis(info, axis_name, shard_dim, mesh_sizes, split_shapes, "gather")
            full_elements = full_info.num_elements()
            forward_elements = full_elements
            materialized_elements = full_elements
            peak_elements = max(peak_elements, full_elements)
        after_info = after_info.with_axis(axis_name, Axis(axis_name, shard_dim))
    elif step.name in {"allgather_forward_split_backward", "allgather_forward_reducescatter_backward"}:
        axis_name, shard_dim = step.args
        after_info = _rescale_axis(info, axis_name, shard_dim, mesh_sizes, split_shapes, "gather")
        after_elements = after_info.num_elements()
        forward_elements = after_elements
        backward_elements = after_elements
        materialized_elements = after_elements
        peak_elements = max(before_elements, after_elements)
        after_info = after_info.with_axis(axis_name, Axis(axis_name))
    elif step.name in {
        "allreduce_forward_identity_backward",
        "identity_forward_allreduce_backward",
        "broadcast_forward_allreduce_backward",
    }:
        forward_elements = before_elements
        backward_elements = before_elements
    elif step.name == "alltoall_repartition":
        source_axis, dest_axis, shard_dim = step.args
        requires_shapes = True
        after_info = _rescale_axis(info, source_axis, shard_dim, mesh_sizes, split_shapes, "gather")
        after_info = _rescale_axis(after_info, dest_axis, shard_dim, mesh_sizes, split_shapes, "split")
        forward_elements = before_elements
        backward_elements = max(before_elements, after_info.num_elements())
        peak_elements = max(before_elements, after_info.num_elements())
        after_info = after_info.with_axis(source_axis, Axis(source_axis)).with_axis(dest_axis, Axis(dest_axis, shard_dim))
    elif step.name == "owner_swap":
        requires_shapes = True
        forward_elements = before_elements
        backward_elements = before_elements

    forward_bytes = forward_elements * info.dtype_size
    backward_bytes = backward_elements * info.dtype_size
    return PlanCost(
        static.score,
        static.collectives,
        static.materializations,
        static.peak_factor,
        forward_bytes,
        backward_bytes,
        peak_elements,
        materialized_elements,
        requires_shapes,
    ), after_info


def _combine_costs(costs):
    score = 0
    collectives = 0
    materializations = 0
    peak_factor = 1
    forward_bytes = 0
    backward_bytes = 0
    peak_elements = 0
    materialized_elements = 0
    requires_shapes = False
    invalid_reason = ""
    for cost in costs:
        score += cost.score
        collectives += cost.collectives
        materializations += cost.materializations
        peak_factor = max(peak_factor, cost.peak_factor)
        forward_bytes += cost.forward_bytes
        backward_bytes += cost.backward_bytes
        peak_elements = max(peak_elements, cost.peak_elements)
        materialized_elements += cost.materialized_elements
        requires_shapes = requires_shapes or cost.requires_shapes
        invalid_reason = invalid_reason or cost.invalid_reason
    return PlanCost(
        score,
        collectives,
        materializations,
        peak_factor,
        forward_bytes,
        backward_bytes,
        peak_elements,
        materialized_elements,
        requires_shapes,
        invalid_reason,
    )


def _apply_policy(cost, policy):
    if policy is False or policy is None:
        return cost
    policy = _coerce_policy(policy)
    return cost.with_score(policy.score(cost))


def estimate_plan_cost(steps, runtime_info=None, mesh_sizes=None, split_shapes=None, policy=None):
    costs = []
    info = runtime_info
    for step in steps:
        cost, info = _dynamic_step_cost(step, info, mesh_sizes, split_shapes)
        costs.append(cost)
    if policy is None:
        policy = resolve_plan_policy() if runtime_info is not None else False
    return _apply_policy(_combine_costs(costs), policy)


def estimate_alternative_cost(alternative, input0_info, input1_info, output_info, mesh_sizes=None, split_shapes=None, policy=None):
    costs = [
        estimate_plan_cost(alternative.input0_steps, input0_info, mesh_sizes, split_shapes, policy=False),
        estimate_plan_cost(alternative.input1_steps, input1_info, mesh_sizes, split_shapes, policy=False),
        estimate_plan_cost(alternative.output_steps, output_info, mesh_sizes, split_shapes, policy=False),
    ]
    if policy is None:
        policy = resolve_plan_policy()
    return _apply_policy(_combine_costs(costs), policy)


def rank_alternatives(
    alternatives,
    input0_info=None,
    input1_info=None,
    output_info=None,
    mesh_sizes=None,
    split_shapes=None,
    policy=None,
    output_infos=None,
):
    if input0_info is not None and input1_info is not None and output_info is not None:
        policy = resolve_plan_policy(policy=policy)
        alternatives = tuple(
            alternative.with_cost(
                estimate_alternative_cost(
                    alternative,
                    input0_info,
                    input1_info,
                    output_infos.get(alternative.name, output_info) if output_infos is not None else output_info,
                    mesh_sizes,
                    split_shapes,
                    policy,
                )
            )
            for alternative in alternatives
        )
    elif input0_info is not None and input1_info is not None and output_infos is not None:
        policy = resolve_plan_policy(policy=policy)
        alternatives = tuple(
            alternative.with_cost(
                estimate_alternative_cost(
                    alternative,
                    input0_info,
                    input1_info,
                    output_infos[alternative.name],
                    mesh_sizes,
                    split_shapes,
                    policy,
                )
            ) if alternative.name in output_infos else alternative
            for alternative in alternatives
        )
        if all(alternative.name in output_infos for alternative in alternatives):
            return _rank_alternatives(alternatives)
        return alternatives
    return _rank_alternatives(alternatives)


def _rank_alternatives(alternatives):
    return tuple(sorted(alternatives, key=lambda alternative: (
        bool(alternative.cost.invalid_reason),
        alternative.cost.score,
        alternative.cost.peak_elements,
        alternative.cost.total_bytes,
        alternative.cost.materializations,
        alternative.cost.collectives,
        alternative.name,
    )))


def _replace_output_steps(output_steps, replacements):
    steps = []
    for step in output_steps:
        steps.extend(replacements.get(step, (step,)))
    return tuple(steps)


def build_binary_transition_plan(input0_spec, input1_spec, output_spec):
    cache_key = ("binary", _spec_cache_key(input0_spec), _spec_cache_key(input1_spec), _spec_cache_key(output_spec))
    cached = _transition_plan_cache.get(cache_key)
    if cached is not None:
        return cached

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

        input_axis_names = {axis.name for axis in state.axes}
        input_shard_dims = {axis.shard_dim for axis in state.axes if axis.shard_dim}
        backward_reduced_dims = {
            step.args[-1]
            for step in steps
            if step.name in {
                "allgather_forward_reducescatter_backward",
                "identity_forward_allreduce_backward",
            }
        }
        for output_axis in output_state.axes:
            if not output_axis.shard_dim or output_axis.name in input_axis_names:
                continue
            if output_axis.shard_dim in reduction_dims or output_axis.shard_dim in input_shard_dims:
                continue
            if output_axis.shard_dim in backward_reduced_dims:
                continue
            steps.append(PlanStep("identity_forward_allreduce_backward", (output_axis.shard_dim,)))
            backward_reduced_dims.add(output_axis.shard_dim)
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

    input0_steps = normalize_input_steps(input0_state)
    input1_steps = normalize_input_steps(input1_state)
    output_steps = tuple(output_steps)
    alternatives = []
    selected_name = "default"
    default_alternative = PlanAlternative(
        selected_name,
        input0_steps,
        input1_steps,
        output_steps,
        tuple(candidates),
        estimate_plan_cost((*input0_steps, *input1_steps, *output_steps)),
    )
    alternatives.append(default_alternative)

    expanded_reductions = {}
    for step in output_steps:
        if step.name != "reducescatter_forward_allgather_backward":
            continue
        axis_name, shard_dim = step.args
        expanded_reductions[step] = (
            PlanStep("allreduce_forward_identity_backward", (shard_dim,)),
            PlanStep("split_forward_allgather_backward", (axis_name, shard_dim)),
        )
    if expanded_reductions:
        expanded_output_steps = _replace_output_steps(output_steps, expanded_reductions)
        alternatives.append(
            PlanAlternative(
                "allreduce_then_split",
                input0_steps,
                input1_steps,
                expanded_output_steps,
                tuple(candidates),
                estimate_plan_cost((*input0_steps, *input1_steps, *expanded_output_steps)),
            )
        )

    if post_repartition is not None:
        optimized_input0_steps = tuple(
            PlanStep("identity_forward_allreduce_backward", (post_repartition.shard_dim,))
            if step.args and step.args[0] == post_repartition.dest_axis else step
            for step in input0_steps
            if not (step.args and step.args[0] == post_repartition.source_axis)
        )
        optimized_input1_steps = tuple(
            PlanStep("identity_forward_allreduce_backward", (post_repartition.shard_dim,))
            if step.args and step.args[0] == post_repartition.dest_axis else step
            for step in input1_steps
            if not (step.args and step.args[0] == post_repartition.source_axis)
        )
        optimized_output_steps = (*output_steps, PlanStep("alltoall_repartition", (
            post_repartition.source_axis,
            post_repartition.dest_axis,
            post_repartition.shard_dim,
        )))
        alternatives.append(
            PlanAlternative(
                "alltoall_repartition",
                optimized_input0_steps,
                optimized_input1_steps,
                optimized_output_steps,
                tuple(candidates),
                estimate_plan_cost((*optimized_input0_steps, *optimized_input1_steps, *optimized_output_steps)),
                reason="requires runtime split-shape validation",
            )
        )

    alternatives = _rank_alternatives(alternatives)
    selected_name = alternatives[0].name

    plan = BinaryTransitionPlan(
        classification=classification,
        input0_steps=input0_steps,
        input1_steps=input1_steps,
        output_steps=output_steps,
        reduction_dims=tuple(reduction_dims),
        scatter_output_by_dim=tuple(scatter_output_by_dim.items()),
        candidates=tuple(candidates),
        alternatives=alternatives,
        top_ranked_alternative=selected_name,
        post_repartition=post_repartition,
    )
    _transition_plan_cache[cache_key] = plan
    return plan


def tensor_spec(axes, partials=()):
    return TensorSpec(Axes(axes), partials)


def local_axis(name):
    return Axis(name)


def sharded_axis(name, shard_dim):
    return Axis(name, shard_dim)
