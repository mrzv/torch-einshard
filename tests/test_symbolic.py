import pytest

from torch_einshard.grammar import parse_sharding
from torch_einshard.symbolic import (
    ExecutionPlan,
    TensorState,
    build_binary_transition_plan,
    build_unary_transition_plan,
    classify_binary,
    classify_unary,
    last_candidates,
    last_plan,
    local_axis,
    set_last_plan,
    sharded_axis,
    tensor_spec,
)


def test_tensor_state_from_spec_tracks_placements_and_partials():
    x, _, _ = parse_sharding("a/dp b c/tp // sp -> a b c")

    state = TensorState.from_spec(x)

    assert [axis.name for axis in state.axes] == ["a", "b", "c"]
    assert state.placement_dict() == {"a": "dp", "b": None, "c": "tp"}
    assert state.partials == ("sp",)


def test_tensor_state_flattens_axis_groups():
    x, _, _ = parse_sharding("a (b/tp c) -> a b/tp c")

    state = TensorState.from_spec(x)

    assert [axis.name for axis in state.axes] == ["a", "b", "c"]
    assert state.placement_dict() == {"a": None, "b": "tp", "c": None}


def test_tensor_state_copies_axes_from_source_spec():
    x, _, _ = parse_sharding("a/dp b -> a b")
    state = TensorState.from_spec(x)

    x.axes[0].name = "mutated"
    x.axes[0].shard_dim = "tp"

    assert [axis.name for axis in state.axes] == ["a", "b"]
    assert state.placement_dict() == {"a": "dp", "b": None}


def test_tensor_state_rejects_duplicate_axis_names():
    x, _, _ = parse_sharding("a a -> a")

    with pytest.raises(ValueError, match="unique axis names"):
        TensorState.from_spec(x)


def test_tensor_state_requires_ellipsis_expansion():
    x, _, _ = parse_sharding("... c -> ... c")

    with pytest.raises(ValueError, match="ellipsis expansion"):
        TensorState.from_spec(x)


def test_unary_classification_reports_placement_and_partial_deltas():
    x, _, z = parse_sharding("a/dp b // tp -> a b/sp // dp")

    classification = classify_unary(x, z)

    assert [(delta.name, delta.source, delta.target) for delta in classification.placement_deltas] == [
        ("a", "dp", None),
        ("b", None, "sp"),
    ]
    assert classification.removed_partials == ("tp",)
    assert classification.added_partials == ("dp",)


def test_unary_classification_rejects_axis_mismatch():
    x, _, z = parse_sharding("a b -> a c")

    with pytest.raises(ValueError, match="axes must match"):
        classify_unary(x, z)


def test_unary_transition_plan_splits_and_gathers():
    x, _, z = parse_sharding("a b -> a/dp b")
    assert build_unary_transition_plan(x, z).names() == ["split_forward_allgather_backward"]

    x, _, z = parse_sharding("a/dp b -> a b")
    assert build_unary_transition_plan(x, z).names() == ["allgather_forward_split_backward"]


def test_unary_transition_plan_handles_partials_before_layout_changes():
    x, _, z = parse_sharding("a b // tp -> a/tp b")

    plan = build_unary_transition_plan(x, z)

    assert [(step.name, step.args) for step in plan.steps] == [
        ("reducescatter_forward_allgather_backward", ("a", "tp")),
    ]


def test_unary_transition_plan_uses_reduce_scatter_backward_for_output_partial():
    x, _, z = parse_sharding("a/tp b -> a b // tp")

    plan = build_unary_transition_plan(x, z)

    assert [(step.name, step.args) for step in plan.steps] == [
        ("allgather_forward_reducescatter_backward", ("a", "tp")),
    ]


def test_unary_transition_plan_adds_partials_and_permutation():
    x, _, z = parse_sharding("b a -> a b // dp")

    plan = build_unary_transition_plan(x, z)

    assert [(step.name, step.args) for step in plan.steps] == [
        ("identity_forward_allreduce_backward", ("dp",)),
        ("permute", (("a", "b"),)),
    ]


def test_binary_classification_separates_axis_roles():
    x, y, z = parse_sharding("b/dp a c/sp, b/dp c/sp d -> b/dp a d")

    classification = classify_binary(x, y, z)

    assert classification.free_axes == ("a", "d")
    assert classification.shared_output_axes == ("b",)
    assert classification.contracted_axes == ("c",)
    assert classification.output_only_axes == ()


def test_binary_classification_reports_output_only_axes():
    x, y, z = parse_sharding("a b, b c -> a c d")

    classification = classify_binary(x, y, z)
    assert classification.output_only_axes == ("d",)


def test_binary_transition_plan_normalizes_contracted_axis_and_allreduces():
    x, y, z = parse_sharding("a b/tp, b c -> a c")

    plan = build_binary_transition_plan(x, y, z)

    assert [step.name for step in plan.input0_steps] == []
    assert [(step.name, step.args) for step in plan.input1_steps] == [
        ("split_forward_allgather_backward", ("b", "tp")),
    ]
    assert [(step.name, step.args) for step in plan.output_steps] == [
        ("rank_local_einsum", ()),
        ("allreduce_forward_identity_backward", ("tp",)),
    ]
    assert plan.reduction_dims == ("tp",)


def test_binary_transition_plan_reduce_scatters_to_output_axis():
    x, y, z = parse_sharding("a b/tp, b c -> a/tp c")

    plan = build_binary_transition_plan(x, y, z)

    assert [(step.name, step.args) for step in plan.output_steps] == [
        ("rank_local_einsum", ()),
        ("reducescatter_forward_allgather_backward", ("a", "tp")),
    ]
    assert plan.scatter_output_by_dim == (("tp", "a"),)


def test_binary_transition_plan_keeps_output_partial_reduction():
    x, y, z = parse_sharding("a b/tp, b c -> a c // tp")

    plan = build_binary_transition_plan(x, y, z)

    assert [(step.name, step.args) for step in plan.output_steps] == [
        ("rank_local_einsum", ()),
    ]


def test_binary_transition_plan_detects_post_repartition():
    x, y, z = parse_sharding("a/tp b, c d -> a c/tp")

    plan = build_binary_transition_plan(x, y, z)

    assert [(step.name, step.args) for step in plan.input0_steps] == [
        ("allgather_forward_reducescatter_backward", ("a", "tp")),
    ]
    assert [(step.name, step.args) for step in plan.input1_steps] == [
        ("split_forward_allgather_backward", ("c", "tp")),
    ]
    assert [(step.name, step.args) for step in plan.output_steps] == [
        ("rank_local_einsum", ()),
    ]
    assert plan.post_repartition.shard_dim == "tp"
    assert [(candidate.name, candidate.args) for candidate in plan.candidates] == [
        ("alltoall_repartition", ("a", "c", "tp")),
    ]
    assert [(step.name, step.args) for step in plan.candidates[0].fallback_steps] == [
        ("allgather_forward_reducescatter_backward", ("a", "tp")),
        ("split_forward_allgather_backward", ("c", "tp")),
    ]


def test_binary_transition_plan_models_owner_swap_atomically():
    x, y, z = parse_sharding("a k/dp m/sp, k/sp m/dp b -> a/sp b/dp")

    plan = build_binary_transition_plan(x, y, z)

    assert [(step.name, step.args) for step in plan.input1_steps] == [
        ("owner_swap", (("sp", "dp"), ("dp", "sp"))),
    ]
    assert [(candidate.name, candidate.args) for candidate in plan.candidates] == [
        ("owner_swap", (("sp", "dp"), ("dp", "sp"))),
    ]


def test_binary_transition_plan_rejects_non_owner_swap_permutation():
    x, y, z = parse_sharding("a k/dp m/sp n/dp, k/sp m/dp n/tp b -> a b")

    with pytest.raises(NotImplementedError, match="owner-swap-compatible"):
        build_binary_transition_plan(x, y, z)


def test_execution_plan_records_steps_and_last_plan_snapshot():
    plan = ExecutionPlan()
    plan.add("split_forward_allgather_backward", "a", "dp")
    plan.add("rank_local_einsum")

    set_last_plan(plan)

    assert plan.names() == ["split_forward_allgather_backward", "rank_local_einsum"]
    assert [step.name for step in last_plan()] == plan.names()
    assert last_candidates() == ()


def test_execution_plan_records_candidate_snapshot():
    plan = ExecutionPlan()
    plan.consider("alltoall_repartition", "a", "b", "dp", status="rejected", reason="missing shapes")

    set_last_plan(plan)

    candidate, = last_candidates()
    assert candidate.name == "alltoall_repartition"
    assert candidate.args == ("a", "b", "dp")
    assert candidate.status == "rejected"
    assert candidate.reason == "missing shapes"


def test_execution_plan_execute_records_step_and_calls_function():
    plan = ExecutionPlan()

    result = plan.execute("double", lambda x: x * 2, 3, step_args=("x",))

    assert result == 6
    assert [(step.name, step.args) for step in plan.snapshot()] == [("double", ("x",))]


def test_tensor_spec_factories_build_specs():
    spec = tensor_spec([local_axis("a"), sharded_axis("b", "tp")], partials=("dp",))

    assert [axis.name for axis in spec.axes] == ["a", "b"]
    assert spec.axes[1].shard_dim == "tp"
    assert spec.partials == ("dp",)
