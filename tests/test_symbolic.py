import pytest

from torch_einshard.grammar import parse_sharding
from torch_einshard.symbolic import (
    ExecutionPlan,
    PlanPolicy,
    PlanStep,
    TensorRuntimeInfo,
    TensorState,
    build_binary_transition_plan,
    build_unary_transition_plan,
    classify_binary,
    classify_unary,
    clear_plan_cache,
    estimate_alternative_cost,
    estimate_plan_cost,
    last_alternatives,
    last_candidates,
    last_plan,
    local_axis,
    get_default_policy,
    get_optimization_policy,
    optimize,
    rank_alternatives,
    resolve_plan_policy,
    set_default_policy,
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


def test_tensor_state_tracks_replicated_mesh_dims():
    x, _, _ = parse_sharding("a/dp b // tp -> a b")

    state = TensorState.from_spec(x, mesh_dim_names=("dp", "sp", "tp"))

    assert state.replicated_dims == ("sp",)


def test_tensor_state_treats_compound_mesh_dim_components_as_used():
    x, _, _ = parse_sharding("a/dp-sp b // tp-sp -> a b")

    state = TensorState.from_spec(x, mesh_dim_names=("dp", "sp", "tp", "pp"))

    assert state.replicated_dims == ("pp",)


def test_tensor_state_excludes_compound_candidate_mesh_dims_with_used_components():
    x, _, _ = parse_sharding("a/sp1 b -> a b")

    state = TensorState.from_spec(x, mesh_dim_names=("sp1-sp2", "tp"))

    assert state.replicated_dims == ("tp",)


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


def test_unary_transition_plan_models_owner_swap_atomically():
    x, _, z = parse_sharding("a/dp b/sp -> a/sp b/dp")

    plan = build_unary_transition_plan(x, z)

    assert [(step.name, step.args) for step in plan.steps] == [
        ("owner_swap", (("dp", "sp"), ("sp", "dp"))),
    ]


def test_unary_transition_plan_rejects_owner_swap_with_unchanged_involved_axis():
    x, _, z = parse_sharding("a/dp b/sp -> a/sp b/sp")

    plan = build_unary_transition_plan(x, z)

    assert [(step.name, step.args) for step in plan.steps] == [
        ("allgather_forward_split_backward", ("a", "dp")),
        ("split_forward_allgather_backward", ("a", "sp")),
    ]


def test_unary_transition_plan_rejects_conflicting_owner_swap_mapping():
    x, _, z = parse_sharding("a/dp b/dp c/sp d/tp -> a/sp b/tp c/dp d/dp")

    plan = build_unary_transition_plan(x, z)

    assert "owner_swap" not in plan.names()


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


def test_transition_plan_cache_reuses_immutable_plans():
    clear_plan_cache()
    x, y, z = parse_sharding("a b/tp, b c -> a/tp c")

    first = build_binary_transition_plan(x, y, z)
    second = build_binary_transition_plan(x, y, z)

    assert second is first

    clear_plan_cache()
    third = build_binary_transition_plan(x, y, z)
    assert third is not first


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
    assert [(alternative.name, alternative.cost.score) for alternative in plan.alternatives] == [
        ("default", 8),
        ("allreduce_then_split", 14),
    ]


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
    assert [alternative.name for alternative in plan.alternatives] == [
        "alltoall_repartition",
        "default",
    ]
    assert plan.alternatives[0].reason == "requires runtime split-shape validation"


def test_runtime_cost_estimate_uses_combined_forward_backward_bytes():
    x, y, z = parse_sharding("a b/tp, b c -> a/tp c")
    plan = build_binary_transition_plan(x, y, z)
    default = next(alternative for alternative in plan.alternatives if alternative.name == "default")
    expanded = next(alternative for alternative in plan.alternatives if alternative.name == "allreduce_then_split")
    input0 = TensorRuntimeInfo(tuple(x.axes), (8, 4), 4)
    input1 = TensorRuntimeInfo(tuple(y.axes), (4, 6), 4)
    output = TensorRuntimeInfo(tuple(z.axes), (8, 6), 4)

    default_cost = estimate_alternative_cost(default, input0, input1, output, {"tp": 2}, policy=PlanPolicy.from_mode("training"))
    expanded_cost = estimate_alternative_cost(expanded, input0, input1, output, {"tp": 2}, policy=PlanPolicy.from_mode("training"))

    assert default_cost.total_bytes == default_cost.forward_bytes + default_cost.backward_bytes
    assert default_cost.total_bytes < expanded_cost.total_bytes
    assert default_cost.score < expanded_cost.score


def test_runtime_cost_estimate_scales_with_dtype_size():
    x, y, z = parse_sharding("a/tp b, c d -> a c/tp")
    plan = build_binary_transition_plan(x, y, z)
    alternative = next(alternative for alternative in plan.alternatives if alternative.name == "alltoall_repartition")
    small_dtype = (
        TensorRuntimeInfo(tuple(x.axes), (4, 5), 2),
        TensorRuntimeInfo(tuple(y.axes), (6, 7), 2),
        TensorRuntimeInfo(tuple(z.axes), (8, 6), 2),
    )
    large_dtype = (
        TensorRuntimeInfo(tuple(x.axes), (4, 5), 8),
        TensorRuntimeInfo(tuple(y.axes), (6, 7), 8),
        TensorRuntimeInfo(tuple(z.axes), (8, 6), 8),
    )

    small_cost = estimate_alternative_cost(alternative, *small_dtype, mesh_sizes={"tp": 2}, policy=False)
    large_cost = estimate_alternative_cost(alternative, *large_dtype, mesh_sizes={"tp": 2}, policy=False)
    large_training_cost = estimate_alternative_cost(
        alternative,
        *large_dtype,
        mesh_sizes={"tp": 2},
        policy=PlanPolicy.from_mode("training"),
    )

    assert large_cost.total_bytes == small_cost.total_bytes * 4
    assert alternative.cost.score == large_cost.score
    assert alternative.cost.score < large_training_cost.score


def test_runtime_cost_estimate_uses_scoped_policy_by_default():
    x, y, z = parse_sharding("a b/tp, b c -> a/tp c")
    plan = build_binary_transition_plan(x, y, z)
    default = next(alternative for alternative in plan.alternatives if alternative.name == "default")
    args = (
        default,
        TensorRuntimeInfo(tuple(x.axes), (8, 4), 4),
        TensorRuntimeInfo(tuple(y.axes), (4, 6), 4),
        TensorRuntimeInfo(tuple(z.axes), (8, 6), 4),
    )

    set_default_policy("communication")
    try:
        default_policy_cost = estimate_alternative_cost(*args, mesh_sizes={"tp": 2})
    finally:
        set_default_policy(None)
    explicit_training_cost = estimate_alternative_cost(
        *args,
        mesh_sizes={"tp": 2},
        policy=PlanPolicy.from_mode("training"),
    )

    assert default_policy_cost.score != explicit_training_cost.score


def test_runtime_cost_estimate_accepts_named_policy_modes():
    x, y, z = parse_sharding("a b/tp, b c -> a/tp c")
    plan = build_binary_transition_plan(x, y, z)
    default = next(alternative for alternative in plan.alternatives if alternative.name == "default")
    args = (
        default,
        TensorRuntimeInfo(tuple(x.axes), (8, 4), 4),
        TensorRuntimeInfo(tuple(y.axes), (4, 6), 4),
        TensorRuntimeInfo(tuple(z.axes), (8, 6), 4),
    )

    memory_cost = estimate_alternative_cost(*args, mesh_sizes={"tp": 2}, policy="memory")
    communication_cost = estimate_alternative_cost(*args, mesh_sizes={"tp": 2}, policy="communication")
    plan_cost = estimate_plan_cost(default.output_steps, args[3], mesh_sizes={"tp": 2}, policy="memory")

    assert memory_cost.score != communication_cost.score
    assert plan_cost.score > default.cost.score


def test_runtime_ranking_costs_alltoall_sets():
    x, y, z = parse_sharding("a/tp b, c d -> a c/tp")
    plan = build_binary_transition_plan(x, y, z)

    ranked = rank_alternatives(
        plan.alternatives,
        TensorRuntimeInfo(tuple(x.axes), (4, 5), 8),
        TensorRuntimeInfo(tuple(y.axes), (6, 7), 8),
        TensorRuntimeInfo(tuple(z.axes), (8, 6), 8),
        mesh_sizes={"tp": 2},
    )

    assert [alternative.name for alternative in ranked] == ["alltoall_repartition", "default"]
    assert all(alternative.cost.forward_bytes > 0 for alternative in ranked)


def test_plan_cost_estimate_models_broadcast_step():
    x, _, _ = parse_sharding("a b -> a b")
    info = TensorRuntimeInfo(tuple(x.axes), (8, 4), 4)

    cost = estimate_plan_cost(
        [PlanStep("broadcast_forward_allreduce_backward", ("dp", 0))],
        info,
        policy=False,
    )
    assert cost.collectives == 1
    assert cost.forward_bytes == 8 * 4 * 4


def test_binary_transition_plan_includes_replicated_gradient_allreduce():
    x, y, z = parse_sharding("a/dp k, k/sp b -> a/dp b/sp")

    plan = build_binary_transition_plan(x, y, z)

    assert [(step.name, step.args) for step in plan.input1_steps] == [
        ("identity_forward_allreduce_backward", ("dp",)),
    ]


def test_uneven_reduce_scatter_cost_models_fallback_materialization():
    x, y, z = parse_sharding("a b/tp, b c -> a/tp c")
    plan = build_binary_transition_plan(x, y, z)
    default = next(alternative for alternative in plan.alternatives if alternative.name == "default")
    info = (
        TensorRuntimeInfo(tuple(x.axes), (5, 8), 4),
        TensorRuntimeInfo(tuple(y.axes), (8, 6), 4),
        TensorRuntimeInfo(tuple(z.axes), (3, 6), 4),
    )

    even = estimate_alternative_cost(
        default,
        *info,
        mesh_sizes={"tp": 2},
        split_shapes={("tp", "a"): (3, 3)},
        policy=False,
    )
    uneven = estimate_alternative_cost(
        default,
        *info,
        mesh_sizes={"tp": 2},
        split_shapes={("tp", "a"): (3, 2)},
        policy=False,
    )

    assert uneven.materialized_elements > even.materialized_elements
    assert uneven.forward_bytes > even.forward_bytes


def test_runtime_ranking_keeps_reduce_scatter_before_allreduce_split():
    x, y, z = parse_sharding("a b/tp, b c -> a/tp c")
    plan = build_binary_transition_plan(x, y, z)
    ranked = rank_alternatives(
        plan.alternatives,
        TensorRuntimeInfo(tuple(x.axes), (16, 8), 4),
        TensorRuntimeInfo(tuple(y.axes), (8, 12), 4),
        TensorRuntimeInfo(tuple(z.axes), (16, 12), 4),
        mesh_sizes={"tp": 4},
    )

    assert [alternative.name for alternative in ranked] == ["default", "allreduce_then_split"]
    assert ranked[0].cost.total_bytes < ranked[1].cost.total_bytes


def test_named_policy_modes_are_resolved():
    assert resolve_plan_policy(optimize="training").mode == "training"
    assert resolve_plan_policy(optimize="memory").mode == "memory"
    assert resolve_plan_policy(optimize="communication").mode == "communication"

    with pytest.raises(ValueError, match="Unknown optimization policy"):
        resolve_plan_policy(optimize="unknown")


def test_policy_argument_takes_explicit_policy_and_rejects_ambiguous_optimize():
    policy = PlanPolicy.from_mode("latency")

    assert resolve_plan_policy(policy=policy) is policy
    with pytest.raises(ValueError, match="either optimize or policy"):
        resolve_plan_policy(optimize="memory", policy=policy)


def test_default_and_scoped_policy_precedence():
    set_default_policy(None)
    assert get_default_policy().mode == "training"
    assert get_optimization_policy().mode == "training"

    set_default_policy("communication")
    try:
        assert get_default_policy().mode == "communication"
        assert get_optimization_policy().mode == "communication"

        with optimize("memory"):
            assert get_optimization_policy().mode == "memory"
            assert resolve_plan_policy().mode == "memory"
            assert resolve_plan_policy(optimize="latency").mode == "latency"
            explicit = PlanPolicy.from_mode("inference")
            assert resolve_plan_policy(policy=explicit) is explicit

            with optimize("training"):
                assert get_optimization_policy().mode == "training"

            assert get_optimization_policy().mode == "memory"

        assert get_optimization_policy().mode == "communication"
    finally:
        set_default_policy(None)


def test_policy_modes_change_runtime_scores_for_safe_alternatives():
    x, y, z = parse_sharding("a b/tp, b c -> a/tp c")
    plan = build_binary_transition_plan(x, y, z)
    args = (
        plan.alternatives,
        TensorRuntimeInfo(tuple(x.axes), (64, 32), 4),
        TensorRuntimeInfo(tuple(y.axes), (32, 48), 4),
        TensorRuntimeInfo(tuple(z.axes), (64, 48), 4),
    )

    training = rank_alternatives(*args, mesh_sizes={"tp": 4}, policy=PlanPolicy.from_mode("training"))
    communication = rank_alternatives(*args, mesh_sizes={"tp": 4}, policy=PlanPolicy.from_mode("communication"))

    assert [alternative.name for alternative in training] == [alternative.name for alternative in communication]
    assert training[0].cost.score != communication[0].cost.score


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


def test_execution_plan_records_ranked_alternatives_snapshot():
    plan = ExecutionPlan()
    x, y, z = parse_sharding("a b/tp, b c -> a/tp c")
    transition_plan = build_binary_transition_plan(x, y, z)

    plan.rank(transition_plan.alternatives, selected="default", reason="validated")
    set_last_plan(plan)

    assert [(alternative.name, alternative.status) for alternative in last_alternatives()] == [
        ("default", "selected"),
        ("allreduce_then_split", "ranked"),
    ]


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
