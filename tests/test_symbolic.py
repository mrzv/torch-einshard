import pytest

from torch_einshard.grammar import parse_sharding
from torch_einshard.symbolic import (
    ExecutionPlan,
    TensorState,
    classify_binary,
    classify_unary,
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


def test_execution_plan_records_steps_and_last_plan_snapshot():
    plan = ExecutionPlan()
    plan.add("split_forward_allgather_backward", "a", "dp")
    plan.add("rank_local_einsum")

    set_last_plan(plan)

    assert plan.names() == ["split_forward_allgather_backward", "rank_local_einsum"]
    assert [step.name for step in last_plan()] == plan.names()


def test_tensor_spec_factories_build_specs():
    spec = tensor_spec([local_axis("a"), sharded_axis("b", "tp")], partials=("dp",))

    assert [axis.name for axis in spec.axes] == ["a", "b"]
    assert spec.axes[1].shard_dim == "tp"
    assert spec.partials == ("dp",)
