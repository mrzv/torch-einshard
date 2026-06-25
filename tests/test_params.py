import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import torch_einshard as es
from torch_einshard.params import (
    PARAM_SPEC_ATTR,
    PARAM_STATE_ATTR,
    parameter_operand_state,
    register_parameter_operand,
    register_parameter_state,
)

from conftest import assert_close


def test_param_spec_parses_layout_and_metadata():
    spec = es.ParamSpec("o/tp c", shared="sp1-sp2", reduce=("sp1-sp2",))

    assert spec.layout == "o/tp c"
    assert spec.spec.axes is spec.axes
    assert spec.axes[0].name == "o"
    assert spec.axes[0].shard_dim == "tp"
    assert spec.axes[1].name == "c"
    assert spec.shared == ("sp1-sp2",)
    assert spec.reduce == ("sp1-sp2",)


def test_param_spec_repr_includes_nondefault_metadata():
    spec = es.ParamSpec("o/tp c", shared="sp", reduce="sp")

    assert repr(spec) == "ParamSpec('o/tp c', shared=('sp',), reduce=('sp',))"


def test_param_spec_equality_ignores_cached_tensor_spec_identity():
    assert es.ParamSpec("o c", shared="sp", reduce="sp") == es.ParamSpec(
        "o c",
        shared="sp",
        reduce="sp",
    )


def test_param_spec_rejects_shared_sharded_axis_overlap():
    try:
        es.ParamSpec("o/tp c", shared="tp")
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected overlapping shared and sharded metadata to fail")


def test_param_spec_rejects_duplicate_shard_dims():
    try:
        es.ParamSpec("o/tp c/tp")
    except ValueError as error:
        assert "same mesh dimension" in str(error)
    else:
        raise AssertionError("Expected duplicate parameter shard dimensions to fail")


def test_param_spec_rejects_overlapping_compound_shard_dims():
    try:
        es.ParamSpec("o/dp-sp c/dp")
    except ValueError as error:
        assert "same mesh dimension" in str(error)
    else:
        raise AssertionError("Expected overlapping compound shard dimensions to fail")


def test_param_spec_rejects_shared_compound_shard_overlap():
    try:
        es.ParamSpec("o/dp-sp c", shared="dp")
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected shared metadata over a compound shard component to fail")


def test_param_shard_dims_reads_specs_from_params():
    param = torch.nn.Parameter(torch.zeros(2, 3))
    spec = es.ParamSpec("o/tp c/sp")
    es.set_param_spec(param, spec)

    assert es.param_shard_dims(spec) == ("tp", "sp")
    assert es.param_shard_dims(param) == ("tp", "sp")


def test_param_shard_dims_accepts_parameter_states():
    param = torch.nn.Parameter(torch.zeros(2, 3))
    state = es.ParameterState.from_spec(es.parse_sharding("o/tp c/sp [param] -> o c")[0])
    es.set_parameter_state(param, state)

    assert es.param_shard_dims(state) == ("tp", "sp")
    assert es.param_shard_dims(param) == ("tp", "sp")


def test_param_shard_dims_requires_attached_spec():
    param = torch.nn.Parameter(torch.zeros(2, 3))

    try:
        es.param_shard_dims(param)
    except ValueError as error:
        assert "ParamSpec" in str(error)
    else:
        raise AssertionError("Expected missing ParamSpec to fail")


def test_iter_param_specs_yields_only_attached_specs():
    module = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    spec = es.ParamSpec("o c")
    es.set_param_spec(module[0].weight, spec)

    entries = list(es.iter_param_specs(module))
    assert len(entries) == 1
    name, param, actual_spec = entries[0]
    assert name == "0.weight"
    assert param is module[0].weight
    assert actual_spec is spec


def test_set_param_spec_attaches_compatible_parameter_state():
    param = torch.nn.Parameter(torch.zeros(2, 3))
    spec = es.ParamSpec("o/tp c", shared="sp1-sp2", reduce="sp1-sp2")

    es.set_param_spec(param, spec)
    state = es.get_parameter_state(param)

    assert state.source == "ParamSpec"
    assert state.spec is spec.spec
    assert state.axes is spec.axes
    assert state.layout_shard_dims == ("tp",)
    assert state.init_sync.mode == "explicit"
    assert state.shared == ("sp1-sp2",)
    assert state.grad_comm.mode == "explicit"
    assert state.grad_comm.backend == "native"
    assert state.grad_comm.schedule == "synchronous"
    assert state.reduce == ("sp1-sp2",)


def test_parameter_state_infers_init_sync_from_mesh_dims_and_annotation():
    _, weight, _ = es.parse_sharding("b c, out/tp c [param, grad=async] -> b out/tp")

    state = es.ParameterState.from_spec(
        weight,
        mesh_dim_names=("tp", "sp1", "sp2"),
        source="formula",
    )

    assert state.source == "formula"
    assert state.layout_shard_dims == ("tp",)
    assert state.init_sync.mode == "inferred"
    assert state.shared == ("sp1", "sp2")
    assert state.grad_comm.mode == "inferred"
    assert state.grad_comm.mesh_dims == ()
    assert state.grad_comm.backend == "native"
    assert state.grad_comm.schedule == "async"
    assert state.grad_comm.pending_inference
    assert state.tensor_state.placement_dict() == {"out": "tp", "c": None}
    assert state.tensor_state.replicated_dims == ("sp1", "sp2")


def test_parameter_state_uses_explicit_annotation_overrides():
    _, weight, _ = es.parse_sharding(
        "b c, out c [param, grad=dp:external, init_sync=none] -> b out"
    )

    state = es.ParameterState.from_spec(weight, mesh_dim_names=("dp", "tp"))

    assert state.layout_shard_dims == ()
    assert state.init_sync.mode == "none"
    assert state.shared == ()
    assert state.grad_comm.mode == "explicit"
    assert state.grad_comm.mesh_dims == ("dp",)
    assert state.grad_comm.backend == "external"
    assert state.reduce == ()


def test_parameter_state_defaults_param_grad_to_pending_inference():
    _, weight, _ = es.parse_sharding("b c, out c [param] -> b out")

    state = es.ParameterState.from_spec(weight)

    assert state.grad_comm.mode == "inferred"
    assert state.grad_comm.backend == "native"
    assert state.grad_comm.pending_inference


def test_parameter_state_preserves_explicit_grad_none():
    _, weight, _ = es.parse_sharding("b c, out c [param, grad=none] -> b out")

    state = es.ParameterState.from_spec(weight)

    assert state.grad_comm.mode == "none"
    assert state.grad_comm.backend == "none"
    assert not state.grad_comm.pending_inference


def test_parameter_state_rejects_duplicate_layout_shard_dims():
    _, weight, _ = es.parse_sharding("b c, out/tp c/tp [param] -> b out")

    try:
        es.ParameterState.from_spec(weight)
    except ValueError as error:
        assert "same mesh dimension" in str(error)
    else:
        raise AssertionError("Expected duplicate parameter shard dimensions to fail")


def test_parameter_state_rejects_overlapping_compound_layout_shard_dims():
    _, weight, _ = es.parse_sharding("b c, out/dp-sp c/sp-dp [param] -> b out")

    try:
        es.ParameterState.from_spec(weight)
    except ValueError as error:
        assert "same mesh dimension" in str(error)
    else:
        raise AssertionError("Expected overlapping compound shard dimensions to fail")


def test_parameter_state_rejects_init_sync_shard_overlap():
    _, weight, _ = es.parse_sharding("b c, out/tp c [param, init_sync=tp] -> b out")

    try:
        es.ParameterState.from_spec(weight)
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected overlapping init_sync and sharded metadata to fail")


def test_parameter_state_rejects_partial_parameter_specs():
    weight, _, _ = es.parse_sharding("w // dp [param] -> w")

    try:
        es.ParameterState.from_spec(weight, mesh_dim_names=("dp", "tp"))
    except ValueError as error:
        assert "axis layout" in str(error)
    else:
        raise AssertionError("Expected partial parameter state to fail")


def test_parameter_state_infers_init_sync_from_compound_shard_dim_components():
    _, weight, _ = es.parse_sharding("b c, out/dp-sp c [param] -> b out")

    state = es.ParameterState.from_spec(weight, mesh_dim_names=("dp", "sp", "tp"))

    assert state.layout_shard_dims == ("dp-sp",)
    assert state.init_sync.mode == "inferred"
    assert state.shared == ("tp",)
    assert state.tensor_state.replicated_dims == ("tp",)


def test_parameter_state_excludes_compound_candidate_init_sync_overlap():
    _, weight, _ = es.parse_sharding("b c, out/sp1 c [param] -> b out")

    state = es.ParameterState.from_spec(weight, mesh_dim_names=("sp1-sp2", "tp"))

    assert state.shared == ("tp",)
    assert state.tensor_state.replicated_dims == ("tp",)


def test_iter_parameter_states_yields_attached_states():
    module = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    state = es.ParameterState.from_spec(es.parse_sharding("o c [param] -> o c")[0])
    es.set_parameter_state(module[1].weight, state)

    entries = list(es.iter_parameter_states(module))

    assert len(entries) == 1
    name, param, actual_state = entries[0]
    assert name == "1.weight"
    assert param is module[1].weight
    assert actual_state is state


def test_einshard_registers_annotated_parameter_operands():
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(4, 3))

    output = es.einshard("b c, o c [param, grad=async] -> b o", x, weight)
    state = es.get_parameter_state(weight)

    assert output.shape == (2, 4)
    assert state.source == "formula"
    assert state.layout_shard_dims == ()
    assert state.init_sync.mode == "none"
    assert state.grad_comm.mode == "none"
    assert not state.grad_comm.pending_inference


def test_einshard_infers_parameter_grad_dims_from_visible_sharded_axes(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard(
        "b/dp c, c [param, grad=async] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.init_sync.mode == "inferred"
    assert state.shared == ("dp", "sp")
    assert state.grad_comm.mode == "inferred"
    assert state.grad_comm.mesh_dims == ("dp",)
    assert state.grad_comm.backend == "native"
    assert state.grad_comm.schedule == "async"
    assert not state.grad_comm.pending_inference


def test_einshard_preserves_explicit_parameter_grad_override(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard(
        "b/dp c, c [param, grad=sp] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.grad_comm.mode == "explicit"
    assert state.grad_comm.mesh_dims == ("sp",)


def test_einshard_infers_external_parameter_grad_obligation(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard(
        "b/dp c, c [param, grad=external] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.grad_comm.mode == "inferred"
    assert state.grad_comm.mesh_dims == ("dp",)
    assert state.grad_comm.backend == "external"
    assert state.reduce == ()


def test_einshard_infers_parameter_grad_dims_from_output_partials(dist_env, mesh_2d):
    weight = torch.nn.Parameter(torch.ones(3))
    input_spec, param_spec, output_spec = es.parse_sharding(
        "b/dp c, c [param, grad=async] -> b/dp c // sp"
    )

    state = parameter_operand_state(
        weight,
        (input_spec, param_spec),
        output_spec,
        1,
        mesh_dim_names=mesh_2d.mesh_dim_names,
    )

    assert state.grad_comm.mesh_dims == ("dp", "sp")


def test_einshard_public_path_defers_output_partial_grad_dims(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard(
        "b c, c [param, grad=async] -> b c // sp",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.grad_comm.pending_inference
    assert state.grad_comm.mesh_dims == ()
    assert state.grad_comm.schedule == "async"


def test_einshard_public_path_defers_output_partial_param_shard_overlap(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard(
        "b c/sp, c/sp [param, grad=async] -> b // sp",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.grad_comm.pending_inference
    assert state.grad_comm.mesh_dims == ()


def test_einshard_defers_unary_output_partial_parameter_grad(dist_env, mesh_2d):
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("c [param, grad=async] -> c // dp", weight, mesh=mesh_2d)
    state = es.get_parameter_state(weight)

    assert state.grad_comm.pending_inference
    assert state.grad_comm.mesh_dims == ()


def test_einshard_defers_inferred_parameter_grad_for_distributed_formula(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard(
        "b/dp c/sp, c [param, grad=async] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.grad_comm.pending_inference
    assert state.grad_comm.mesh_dims == ()


def test_reduce_grad_rejects_pending_native_parameter_grad(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard(
        "b/dp c/sp, c [param, grad=async] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    weight.grad = torch.ones_like(weight)

    try:
        es.reduce_grad_(weight, es.get_parameter_state(weight), mesh_2d)
    except ValueError as error:
        assert "pending inference" in str(error)
    else:
        raise AssertionError("Expected pending gradient communication to fail")


def test_einshard_distributed_pending_grad_fills_nonexplicit_none(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("b c, c [param] -> b c", x, weight)
    es.einshard(
        "b/dp c/sp, c [param, grad=async] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.grad_comm.pending_inference
    assert state.grad_comm.mesh_dims == ()


def test_einshard_distributed_pending_grad_overrides_prior_inferred_concrete(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("b/dp c, c [param, grad=async] -> b/dp c", x, weight, mesh=mesh_2d)
    assert es.get_parameter_state(weight).grad_comm.mesh_dims == ("dp",)

    es.einshard(
        "b/dp c/sp, c [param, grad=async] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.grad_comm.pending_inference
    assert state.grad_comm.mesh_dims == ()
    assert state.grad_comm.schedule == "async"


def test_einshard_prior_pending_grad_masks_later_inferred_concrete(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard(
        "b/dp c/sp, c [param, grad=async] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    es.einshard("b/dp c, c [param, grad=async] -> b/dp c", x, weight, mesh=mesh_2d)
    state = es.get_parameter_state(weight)

    assert state.grad_comm.pending_inference
    assert state.grad_comm.mesh_dims == ()
    assert state.grad_comm.schedule == "async"


def test_einshard_distributed_pending_grad_conflicts_with_explicit_none(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("b c, c [param, grad=none] -> b c", x, weight)

    try:
        es.einshard(
            "b/dp c/sp, c [param, grad=async] -> b/dp c",
            x,
            weight,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected explicit gradient opt-out conflict to fail")

    assert es.get_parameter_state(weight).explicit_grad_comm_none


def test_einshard_pending_grad_backend_conflicts_with_existing_native_reduce(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_param_spec(weight, es.ParamSpec("c", reduce="sp"))

    try:
        es.einshard(
            "b/dp c/sp, c [param, grad=external] -> b/dp c",
            x,
            weight,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected native/external gradient backend conflict to fail")

    assert es.get_parameter_state(weight).grad_comm.backend == "native"


def test_einshard_pending_external_grad_conflicts_with_later_native_inference(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard(
        "b/dp c/sp, c [param, grad=external] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )

    try:
        es.einshard(
            "b/dp c, c [param, grad=async] -> b/dp c",
            x,
            weight,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected external/native gradient backend conflict to fail")

    assert es.get_parameter_state(weight).grad_comm.backend == "external"


def test_einshard_later_async_annotation_refines_default_schedule(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("b/dp c, c [param] -> b/dp c", x, weight, mesh=mesh_2d)
    assert es.get_parameter_state(weight).grad_comm.schedule == "backend_default"

    es.einshard(
        "b/dp c, c [param, grad=async] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.grad_comm.mesh_dims == ("dp",)
    assert state.grad_comm.schedule == "async"


def test_register_parameter_operand_can_defer_grad_inference(dist_env, mesh_2d):
    weight = torch.nn.Parameter(torch.ones(3))
    input_spec, param_spec, output_spec = es.parse_sharding(
        "b/dp c/sp, c [param, grad=async] -> b/dp c"
    )

    returned = register_parameter_operand(
        weight,
        (input_spec, param_spec),
        output_spec,
        1,
        mesh_dim_names=mesh_2d.mesh_dim_names,
        infer_grad=False,
    )
    state = es.get_parameter_state(weight)

    assert returned is weight
    assert state.grad_comm.pending_inference


def test_einshard_merges_formula_grad_into_layout_only_param_spec(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_param_spec(weight, es.ParamSpec("c"))

    es.einshard(
        "b/dp c, c [param, grad=async] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.source == "ParamSpec+formula"
    assert state.shared == ("dp", "sp")
    assert state.tensor_state.replicated_dims == ("dp", "sp")
    assert state.grad_comm.mesh_dims == ("dp",)
    assert state.grad_comm.schedule == "async"


def test_einshard_accepts_semantically_matching_param_spec_metadata(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_param_spec(weight, es.ParamSpec("c", shared=("dp", "sp"), reduce="sp"))

    es.einshard(
        "b/dp c, c [param, grad=sp] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.source == "ParamSpec+formula"
    assert state.shared == ("dp", "sp")
    assert state.reduce == ("sp",)
    assert state.tensor_state.replicated_dims == ("dp", "sp")


def test_einshard_rejects_param_spec_formula_grad_conflict(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_param_spec(weight, es.ParamSpec("c", reduce="sp"))

    try:
        es.einshard(
            "b/dp c, c [param, grad=async] -> b/dp c",
            x,
            weight,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected ParamSpec/formula gradient conflict to fail")

    assert es.get_parameter_state(weight).source == "ParamSpec"
    assert es.get_parameter_state(weight).reduce == ("sp",)


def test_einshard_rejects_param_spec_formula_init_sync_none_conflict():
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_param_spec(weight, es.ParamSpec("c", shared="sp"))

    try:
        es.einshard("b c, c [param, init_sync=none] -> b c", x, weight)
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected ParamSpec/formula init-sync conflict to fail")

    assert es.get_parameter_state(weight).source == "ParamSpec"
    assert es.get_parameter_state(weight).shared == ("sp",)


def test_einshard_rejects_param_spec_formula_grad_none_conflict():
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_param_spec(weight, es.ParamSpec("c", reduce="sp"))

    try:
        es.einshard("b c, c [param, grad=none] -> b c", x, weight)
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected ParamSpec/formula gradient conflict to fail")

    assert es.get_parameter_state(weight).source == "ParamSpec"
    assert es.get_parameter_state(weight).reduce == ("sp",)


def test_einshard_fills_nonexplicit_formula_none_from_later_inference(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("b c, c [param] -> b c", x, weight)
    assert es.get_parameter_state(weight).grad_comm.mode == "none"

    es.einshard(
        "b/dp c, c [param, grad=async] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    state = es.get_parameter_state(weight)

    assert state.source == "formula"
    assert state.shared == ("dp", "sp")
    assert state.grad_comm.mesh_dims == ("dp",)
    assert state.grad_comm.schedule == "async"


def test_einshard_later_light_formula_does_not_downgrade_tensor_state(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard(
        "b/dp c, c [param, grad=async] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )
    es.einshard("b c, c [param] -> b c", x, weight)
    state = es.get_parameter_state(weight)

    assert state.shared == ("dp", "sp")
    assert state.tensor_state.replicated_dims == ("dp", "sp")
    assert state.grad_comm.mesh_dims == ("dp",)


def test_einshard_preserves_explicit_formula_grad_none_as_conflict(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("b c, c [param, grad=none] -> b c", x, weight)

    try:
        es.einshard(
            "b/dp c, c [param, grad=async] -> b/dp c",
            x,
            weight,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected explicit gradient opt-out conflict to fail")

    assert es.get_parameter_state(weight).grad_comm.mode == "none"


def test_einshard_preserves_explicit_grad_none_through_intermediate_merge(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("b c, c [param, grad=none] -> b c", x, weight)
    es.einshard("b c, c [param] -> b c", x, weight, mesh=mesh_2d)

    try:
        es.einshard(
            "b/dp c, c [param, grad=async] -> b/dp c",
            x,
            weight,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected explicit gradient opt-out conflict to fail")

    assert es.get_parameter_state(weight).explicit_grad_comm_none


def test_einshard_preserves_explicit_formula_init_sync_none_as_conflict(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("b c, c [param, init_sync=none] -> b c", x, weight)

    try:
        es.einshard(
            "b/dp c, c [param, grad=async] -> b/dp c",
            x,
            weight,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected explicit init-sync opt-out conflict to fail")

    assert es.get_parameter_state(weight).init_sync.mode == "none"


def test_einshard_preserves_explicit_init_sync_none_through_intermediate_merge(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("b c, c [param, init_sync=none] -> b c", x, weight)
    es.einshard("b c, c [param, grad=sp] -> b c", x, weight)

    try:
        es.einshard(
            "b/dp c, c [param, grad=sp] -> b/dp c",
            x,
            weight,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected explicit init-sync opt-out conflict to fail")

    assert es.get_parameter_state(weight).explicit_init_sync_none


def test_einshard_reuses_matching_registered_parameter_state(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    for _ in range(2):
        es.einshard(
            "b/dp c, c [param, grad=async] -> b/dp c",
            x,
            weight,
            mesh=mesh_2d,
        )

    assert es.get_parameter_state(weight).grad_comm.mesh_dims == ("dp",)


def test_einshard_rejects_conflicting_registered_parameter_layout(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("b c, c [param] -> b c", x, weight)

    try:
        es.einshard("b c, c/sp [param] -> b c", x, weight, mesh=mesh_2d)
    except ValueError as error:
        assert "different layout" in str(error)
    else:
        raise AssertionError("Expected conflicting parameter layout to fail")


def test_einshard_rejects_annotated_non_parameter_operand():
    x = torch.ones(2, 3)
    scale = torch.ones(3)

    try:
        es.einshard("b c, c [param] -> b c", x, scale)
    except TypeError as error:
        assert "torch.nn.Parameter" in str(error)
    else:
        raise AssertionError("Expected annotated non-Parameter operand to fail")


def test_einshard_failed_dispatch_does_not_attach_parameter_state():
    x = torch.ones(2, 4)
    weight = torch.nn.Parameter(torch.ones(3))

    try:
        es.einshard("b c, c [param] -> b c", x, weight)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected shape mismatch to fail")

    assert es.get_parameter_state(weight) is None


def test_einshard_validates_metadata_conflict_before_dispatch(dist_env, mesh_2d):
    x = torch.ones(2, 4)
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_param_spec(weight, es.ParamSpec("c", reduce="sp"))

    try:
        es.einshard(
            "b/dp c, c [param, grad=async] -> b/dp c",
            x,
            weight,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected metadata conflict to fail before dispatch")

    assert es.get_parameter_state(weight).source == "ParamSpec"


def test_einshard_registration_conflict_does_not_partially_attach_state(dist_env, mesh_2d):
    left = torch.nn.Parameter(torch.ones(2, 3))
    right = torch.nn.Parameter(torch.ones(3))
    es.set_param_spec(right, es.ParamSpec("c", reduce="sp"))

    try:
        es.einshard(
            "b/dp c [param], c [param, grad=async] -> b/dp c",
            left,
            right,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected conflicting parameter metadata to fail")

    assert es.get_parameter_state(left) is None
    assert es.get_parameter_state(right).source == "ParamSpec"


def test_einshard_failed_atomic_registration_does_not_lazily_attach_legacy_state(dist_env, mesh_2d):
    left = torch.nn.Parameter(torch.ones(2, 3))
    right = torch.nn.Parameter(torch.ones(3))
    setattr(right, PARAM_SPEC_ATTR, es.ParamSpec("c", reduce="sp"))

    try:
        es.einshard(
            "b/dp c [param], c [param, grad=async] -> b/dp c",
            left,
            right,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected conflicting parameter metadata to fail")

    assert getattr(left, PARAM_STATE_ATTR, None) is None
    assert getattr(right, PARAM_STATE_ATTR, None) is None


def test_register_parameter_state_failure_does_not_lazily_attach_legacy_state():
    param = torch.nn.Parameter(torch.ones(3))
    setattr(param, PARAM_SPEC_ATTR, es.ParamSpec("c", reduce="sp"))
    spec, _, _ = es.parse_sharding("c [param, grad=dp] -> c")
    state = es.ParameterState.from_spec(spec, source="formula")

    try:
        register_parameter_state(param, state)
    except ValueError as error:
        assert "incompatible metadata" in str(error)
    else:
        raise AssertionError("Expected conflicting parameter metadata to fail")

    assert getattr(param, PARAM_STATE_ATTR, None) is None


def test_einshard_ignores_unnamed_mesh_for_unannotated_local_operation():
    class UnnamedMesh:
        mesh_dim_names = None

    x = torch.ones(2, 3)

    assert_close(es.einshard("b c -> b c", x, mesh=UnnamedMesh()), x)


def test_param_local_slices_uses_mesh_coordinates(dist_env, mesh_2d):
    spec = es.ParamSpec("o/dp c/sp")
    global_shape = (5, 7)
    coord = (mesh_2d.mesh == dist.get_rank()).nonzero()[0].tolist()
    dp_sections = es.helpers.compute_split_shapes_for_factors(
        global_shape[0], mesh_2d.mesh.shape[0], 1
    )
    sp_sections = es.helpers.compute_split_shapes_for_factors(
        global_shape[1], mesh_2d.mesh.shape[1], 1
    )
    expected = (
        slice(sum(dp_sections[:coord[0]]), sum(dp_sections[:coord[0] + 1])),
        slice(sum(sp_sections[:coord[1]]), sum(sp_sections[:coord[1] + 1])),
    )

    assert es.param_local_slices(spec, global_shape, mesh_2d) == expected
    assert es.param_local_shape(spec, global_shape, mesh_2d) == (
        dp_sections[coord[0]],
        sp_sections[coord[1]],
    )


def test_param_local_slices_supports_factor_aware_splits(dist_env, mesh_2d):
    spec = es.ParamSpec("o/dp c")
    global_shape = (10, 3)
    coord = (mesh_2d.mesh == dist.get_rank()).nonzero()[0].tolist()
    sections = es.helpers.compute_split_shapes_for_factors(
        global_shape[0], mesh_2d.mesh.shape[0], 4
    )

    assert es.param_local_slices(spec, global_shape, mesh_2d, factors={"o": 4}) == (
        slice(sum(sections[:coord[0]]), sum(sections[:coord[0] + 1])),
        slice(None),
    )
    assert es.param_local_shape(spec, global_shape, mesh_2d, factors={"o": 4}) == (
        sections[coord[0]],
        3,
    )


def test_param_local_slices_accepts_attached_params(dist_env, mesh_2d):
    param = torch.nn.Parameter(torch.zeros(2, 3))
    es.set_param_spec(param, es.ParamSpec("o/dp c"))

    assert es.param_local_slices(param, (2, 3), mesh_2d)[1] == slice(None)


def test_param_local_slices_accepts_attached_parameter_states(dist_env, mesh_2d):
    param = torch.nn.Parameter(torch.zeros(2, 3))
    state = es.ParameterState.from_spec(es.parse_sharding("o/dp c [param] -> o c")[0])
    es.set_parameter_state(param, state)

    assert es.param_local_slices(state, (2, 3), mesh_2d) == es.param_local_slices(
        es.ParamSpec("o/dp c"),
        (2, 3),
        mesh_2d,
    )
    assert es.param_local_slices(param, (2, 3), mesh_2d)[1] == slice(None)


def test_param_local_slices_rejects_rank_mismatch(dist_env, mesh_2d):
    spec = es.ParamSpec("o/dp c")

    try:
        es.param_local_slices(spec, (2,), mesh_2d)
    except ValueError as error:
        assert "rank" in str(error)
    else:
        raise AssertionError("Expected rank mismatch to fail")


def test_param_local_slices_rejects_missing_mesh_dim(dist_env, mesh_2d):
    spec = es.ParamSpec("o/tp c")

    try:
        es.param_local_slices(spec, (2, 3), mesh_2d)
    except ValueError as error:
        assert "tp" in str(error)
    else:
        raise AssertionError("Expected missing mesh dimension to fail")


def test_param_local_slices_rejects_raw_mesh_compound_group(dist_env, mesh_2d):
    spec = es.ParamSpec("o/dp-sp c")

    try:
        es.param_local_slices(spec, (2, 3), mesh_2d)
    except ValueError as error:
        assert "wrap_mesh" in str(error)
    else:
        raise AssertionError("Expected raw DeviceMesh compound group to fail")


def test_param_local_slices_requires_initialized_process_group(monkeypatch, mesh_2d):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)

    try:
        es.param_local_slices(es.ParamSpec("o/dp c"), (2, 3), mesh_2d)
    except RuntimeError as error:
        assert "initialized process group" in str(error)
    else:
        raise AssertionError("Expected missing process group to fail")


def test_param_shard_metadata_supports_compound_groups(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    spec = es.ParamSpec("o/dp-sp c")
    global_shape = (dist_env.world_size + 3, 2)
    group = mesh["dp-sp"].get_group()
    rank = dist.get_rank(group)
    sections = es.helpers.compute_split_shapes_for_factors(
        global_shape[0], dist.get_world_size(group), 1
    )

    metadata = es.param_shard_metadata(spec, global_shape, mesh)

    assert metadata.global_shape == global_shape
    assert metadata.local_slices == (
        slice(sum(sections[:rank]), sum(sections[:rank + 1])),
        slice(None),
    )
    assert metadata.local_shape == (sections[rank], 2)
    assert metadata.shard_dims == ("dp-sp",)


def test_param_shard_metadata_normalizes_compound_group_order(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    global_shape = (dist_env.world_size + 3, 2)

    dp_sp = es.param_shard_metadata(es.ParamSpec("o/dp-sp c"), global_shape, mesh)
    sp_dp = es.param_shard_metadata(es.ParamSpec("o/sp-dp c"), global_shape, mesh)

    assert dp_sp.local_slices == sp_dp.local_slices
    assert dp_sp.local_shape == sp_dp.local_shape


def test_param_shard_metadata_accepts_parameter_states(dist_env, mesh_2d):
    global_shape = (dist_env.world_size + 3, 2)
    state = es.ParameterState.from_spec(es.parse_sharding("o/dp c [param] -> o c")[0])

    metadata = es.param_shard_metadata(state, global_shape, mesh_2d)
    expected = es.param_shard_metadata(es.ParamSpec("o/dp c"), global_shape, mesh_2d)

    assert metadata == expected


def test_param_local_slices_rejects_sharded_factored_axes(dist_env, mesh_2d):
    spec = es.ParamSpec("(a/dp b) c")

    try:
        es.param_local_slices(spec, (6, 2), mesh_2d)
    except NotImplementedError as error:
        assert "factored-axis" in str(error)
    else:
        raise AssertionError("Expected sharded factored-axis metadata to fail")


def test_sync_param_broadcasts_shared_values(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    spec = es.ParamSpec("o c", shared="dp-sp")
    param = torch.nn.Parameter(torch.full((2, 3), float(dist.get_rank() + 1)))

    es.sync_param_(param, spec, mesh)

    assert_close(param, torch.ones_like(param))


def test_module_param_helpers_use_attached_specs(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    module = nn.Linear(3, 2, bias=False)
    es.set_param_spec(module.weight, es.ParamSpec("o c", shared="dp-sp", reduce="dp-sp"))
    module.weight.data.fill_(float(dist.get_rank() + 1))

    assert es.get_param_spec(module.weight).shared == ("dp-sp",)
    assert es.sync_module_params_(module, mesh) is module
    assert_close(module.weight, torch.ones_like(module.weight))

    module.weight.grad = torch.full_like(module.weight, float(dist.get_rank() + 1))
    es.reduce_module_grads_(module, mesh)
    expected = float(dist.get_world_size() * (dist.get_world_size() + 1) // 2)
    assert_close(module.weight.grad, torch.full_like(module.weight, expected))


def test_module_param_helpers_use_attached_parameter_states(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    module = nn.Linear(3, 2, bias=False)
    state = es.ParameterState.from_spec(
        es.parse_sharding("o c [param, init_sync=dp-sp, grad=dp-sp] -> o c")[0]
    )
    es.set_parameter_state(module.weight, state)
    module.weight.data.fill_(float(dist.get_rank() + 1))

    assert es.sync_module_params_(module, mesh) is module
    assert_close(module.weight, torch.ones_like(module.weight))

    module.weight.grad = torch.full_like(module.weight, float(dist.get_rank() + 1))
    es.reduce_module_grads_(module, mesh)
    expected = float(dist.get_world_size() * (dist.get_world_size() + 1) // 2)
    assert_close(module.weight.grad, torch.full_like(module.weight, expected))


def test_reduce_grad_allreduces_reduce_groups(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    spec = es.ParamSpec("o c", reduce="dp-sp")
    param = torch.nn.Parameter(torch.zeros(2, 3))
    param.grad = torch.full_like(param, float(dist.get_rank() + 1))
    world_size = dist.get_world_size()

    es.reduce_grad_(param, spec, mesh)

    expected = float(world_size * (world_size + 1) // 2)
    assert_close(param.grad, torch.full_like(param, expected))


def test_reduce_grad_skips_external_and_ddp_backends(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    for annotation in ("grad=dp-sp:external", "grad=dp-sp:ddp"):
        state = es.ParameterState.from_spec(
            es.parse_sharding(f"o c [param, {annotation}] -> o c")[0]
        )
        param = torch.nn.Parameter(torch.zeros(2, 3))
        param.grad = torch.full_like(param, float(dist.get_rank() + 1))

        es.reduce_grad_(param, state, mesh)

        assert_close(param.grad, torch.full_like(param, float(dist.get_rank() + 1)))


def test_reduce_grad_allows_missing_grad(dist_env, mesh_2d):
    spec = es.ParamSpec("o c", reduce="dp")
    param = torch.nn.Parameter(torch.zeros(2, 3))

    assert es.reduce_grad_(param, spec, mesh_2d) is param
    assert param.grad is None


def test_ddp_grad_reduction_hook_uses_param_specs(dist_env, mesh_2d):
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    es.set_param_spec(model.weight, es.ParamSpec("o c", reduce="sp"))
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())
    es.register_grad_reduction_hook_(ddp, mesh_2d, ddp_group="dp")

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    ddp(x).sum().backward()

    dp_size = dist.get_world_size(mesh_2d["dp"].get_group())
    sp_size = dist.get_world_size(mesh_2d["sp"].get_group())
    sp_rank = dist.get_rank(mesh_2d["sp"].get_group())
    expected = 0.0
    for peer_sp_rank in range(sp_size):
        expected += 1.0 + peer_sp_rank + sp_size * (dp_size - 1) / 2

    assert sp_rank < sp_size
    assert_close(model.weight.grad, torch.full_like(model.weight.grad, expected))


def test_ddp_grad_reduction_hook_uses_parameter_states(dist_env, mesh_2d):
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    state = es.ParameterState.from_spec(es.parse_sharding("o c [param, grad=sp] -> o c")[0])
    es.set_parameter_state(model.weight, state)
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())
    es.register_grad_reduction_hook_(ddp, mesh_2d, ddp_group="dp")

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    ddp(x).sum().backward()

    dp_size = dist.get_world_size(mesh_2d["dp"].get_group())
    sp_size = dist.get_world_size(mesh_2d["sp"].get_group())
    expected = 0.0
    for peer_sp_rank in range(sp_size):
        expected += 1.0 + peer_sp_rank + sp_size * (dp_size - 1) / 2

    assert_close(model.weight.grad, torch.full_like(model.weight.grad, expected))


def test_ddp_grad_reduction_hook_combines_uniform_reduce_specs(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    es.set_param_spec(model.weight, es.ParamSpec("o c", reduce="sp"))
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())
    es.register_grad_reduction_hook_(
        ddp,
        mesh,
        ddp_group="dp",
        combined_reduce_group="dp-sp",
        combined_reduce="sp",
    )

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    ddp(x).sum().backward()

    dp_size = dist.get_world_size(mesh_2d["dp"].get_group())
    sp_size = dist.get_world_size(mesh_2d["sp"].get_group())
    sp_rank = dist.get_rank(mesh_2d["sp"].get_group())
    expected = 0.0
    for peer_sp_rank in range(sp_size):
        expected += 1.0 + peer_sp_rank + sp_size * (dp_size - 1) / 2

    assert_close(model.weight.grad, torch.full_like(model.weight.grad, expected))


def test_ddp_grad_reduction_hook_combines_uniform_parameter_states(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    state = es.ParameterState.from_spec(es.parse_sharding("o c [param, grad=sp] -> o c")[0])
    es.set_parameter_state(model.weight, state)
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())
    es.register_grad_reduction_hook_(
        ddp,
        mesh,
        ddp_group="dp",
        combined_reduce_group="dp-sp",
        combined_reduce="sp",
    )

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    ddp(x).sum().backward()

    dp_size = dist.get_world_size(mesh_2d["dp"].get_group())
    sp_size = dist.get_world_size(mesh_2d["sp"].get_group())
    expected = 0.0
    for peer_sp_rank in range(sp_size):
        expected += 1.0 + peer_sp_rank + sp_size * (dp_size - 1) / 2

    assert_close(model.weight.grad, torch.full_like(model.weight.grad, expected))


def test_ddp_grad_reduction_hook_combined_option_falls_back_for_mixed_specs(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    model = nn.Linear(1, 1, bias=True)
    model.weight.data.fill_(1.0)
    model.bias.data.zero_()
    es.set_param_spec(model.weight, es.ParamSpec("o c", reduce="sp"))
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())
    es.register_grad_reduction_hook_(
        ddp,
        mesh,
        ddp_group="dp",
        combined_reduce_group="dp-sp",
        combined_reduce="sp",
    )

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    ddp(x).sum().backward()

    dp_size = dist.get_world_size(mesh_2d["dp"].get_group())
    sp_size = dist.get_world_size(mesh_2d["sp"].get_group())
    expected_weight = 0.0
    for peer_sp_rank in range(sp_size):
        expected_weight += 1.0 + peer_sp_rank + sp_size * (dp_size - 1) / 2

    assert_close(model.weight.grad, torch.full_like(model.weight.grad, expected_weight))
    assert_close(model.bias.grad, torch.ones_like(model.bias.grad))


def test_ddp_grad_reduction_hook_validates_combined_reduce_args(mesh_2d):
    model = nn.Linear(1, 1, bias=False)
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())

    try:
        es.register_grad_reduction_hook_(ddp, mesh_2d, combined_reduce_group="dp-sp")
    except ValueError as error:
        assert "combined_reduce" in str(error)
    else:
        raise AssertionError("Expected missing combined_reduce to fail")
