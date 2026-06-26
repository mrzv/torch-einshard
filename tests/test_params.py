from dataclasses import replace

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


def test_param_spec_rejects_overlapping_shared_groups():
    cases = (
        ("dp", "dp"),
        ("dp", "dp-sp"),
    )
    for shared in cases:
        try:
            es.ParamSpec("o c", shared=shared)
        except ValueError as error:
            assert "init_sync" in str(error)
            assert "overlap" in str(error)
        else:
            raise AssertionError("Expected overlapping shared groups to fail")


def test_param_spec_rejects_reduce_sharded_axis_overlap():
    try:
        es.ParamSpec("o/sp c", reduce="sp")
    except ValueError as error:
        assert "grad" in str(error)
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected gradient reduction over a parameter shard dim to fail")


def test_param_spec_rejects_reduce_compound_sharded_axis_overlap():
    try:
        es.ParamSpec("o/dp-sp c", reduce="sp")
    except ValueError as error:
        assert "grad" in str(error)
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected gradient reduction over a compound shard component to fail")


def test_param_spec_rejects_overlapping_reduce_groups():
    cases = (
        ("dp", "dp-sp"),
        ("dp-sp", "sp-dp"),
    )
    for reduce in cases:
        try:
            es.ParamSpec("o c", reduce=reduce)
        except ValueError as error:
            assert "grad" in str(error)
            assert "overlap" in str(error)
        else:
            raise AssertionError("Expected overlapping gradient reduction groups to fail")


def test_param_spec_rejects_repeated_compound_group_components():
    cases = (
        {"layout": "o/dp-dp c"},
        {"layout": "o c", "shared": "dp-dp"},
        {"layout": "o c", "reduce": "dp-dp"},
    )
    for kwargs in cases:
        try:
            es.ParamSpec(**kwargs)
        except ValueError as error:
            assert "repeated mesh" in str(error)
        else:
            raise AssertionError("Expected repeated compound group component to fail")


def test_param_spec_rejects_empty_compound_group_components():
    cases = (
        {"layout": "o c", "shared": ""},
        {"layout": "o c", "reduce": ""},
        {"layout": "o c", "shared": "dp-"},
        {"layout": "o c", "reduce": "dp-"},
    )
    for kwargs in cases:
        try:
            es.ParamSpec(**kwargs)
        except ValueError as error:
            assert "empty mesh" in str(error)
        else:
            raise AssertionError("Expected empty compound group component to fail")


def test_param_spec_rejects_non_string_layout():
    try:
        es.ParamSpec(None)
    except TypeError as error:
        assert "layout" in str(error)
    else:
        raise AssertionError("Expected non-string ParamSpec layout to fail")


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


def test_parameter_state_rejects_grad_shard_overlap():
    _, weight, _ = es.parse_sharding("b c, out/tp c [param, grad=tp] -> b out")

    try:
        es.ParameterState.from_spec(weight)
    except ValueError as error:
        assert "grad" in str(error)
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected overlapping grad and sharded metadata to fail")


def test_parameter_state_rejects_ddp_grad_shard_overlap():
    _, weight, _ = es.parse_sharding("b c, out/tp c [param, grad=tp:ddp] -> b out")

    try:
        es.ParameterState.from_spec(weight)
    except ValueError as error:
        assert "grad" in str(error)
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected DDP grad over a sharded parameter dimension to fail")


def test_parameter_state_allows_external_grad_shard_overlap():
    _, weight, _ = es.parse_sharding("b c, out/tp c [param, grad=tp:external] -> b out")

    state = es.ParameterState.from_spec(weight)

    assert state.grad_comm.backend == "external"
    assert state.reduce == ()


def test_parameter_state_allows_pending_grad_for_sharded_layout():
    _, weight, _ = es.parse_sharding("b c, out/tp c [param, grad=async] -> b out")

    state = es.ParameterState.from_spec(weight)

    assert state.grad_comm.pending_inference


def test_einshard_rejects_inferred_grad_shard_overlap(mesh_tp):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(4, 3))

    try:
        es.einshard(
            "b/tp c, out/tp c [param, grad=async] -> b/tp out/tp",
            x,
            weight,
            mesh=mesh_tp,
        )
    except ValueError as error:
        assert "grad" in str(error)
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected inferred gradient overlap with layout shard dim to fail")

    assert es.get_parameter_state(weight) is None


def test_einshard_rejects_inferred_ddp_grad_shard_overlap(mesh_tp):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(4, 3))

    try:
        es.einshard(
            "b/tp c, out/tp c [param, grad=ddp] -> b/tp out/tp",
            x,
            weight,
            mesh=mesh_tp,
        )
    except ValueError as error:
        assert "grad" in str(error)
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected inferred DDP gradient overlap with layout shard dim to fail")

    assert es.get_parameter_state(weight) is None


def test_einshard_rejects_inferred_grad_compound_shard_overlap(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(4, 3))

    try:
        es.einshard(
            "b/sp c, out/dp-sp c [param, grad=async] -> b/sp out/dp-sp",
            x,
            weight,
            mesh=mesh_2d,
        )
    except ValueError as error:
        assert "grad" in str(error)
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected inferred gradient overlap with compound shard dim to fail")

    assert es.get_parameter_state(weight) is None


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


def test_parameter_state_rejects_layout_component_of_hyphenated_mesh_name():
    _, weight, _ = es.parse_sharding("b c, out/sp1 c [param] -> b out")

    try:
        es.ParameterState.from_spec(weight, mesh_dim_names=("sp1-sp2", "tp"))
    except ValueError as error:
        assert "unknown mesh" in str(error)
    else:
        raise AssertionError("Expected layout component of hyphenated mesh name to fail")


def test_parameter_state_from_layout_infers_init_sync_without_grad_obligation():
    state = es.ParameterState.from_layout(
        "out/tp in",
        mesh_dim_names=("tp", "sp"),
        source="manual",
    )

    assert state.source == "manual"
    assert state.layout_shard_dims == ("tp",)
    assert state.shared == ("sp",)
    assert state.grad_comm.mode == "none"
    assert not state.explicit_grad_comm_none
    assert state.tensor_state.placement_dict() == {"out": "tp", "in": None}
    assert state.tensor_state.replicated_dims == ("sp",)


def test_parameter_state_from_layout_accepts_string_mesh_dim_names():
    class Mesh:
        mesh_dim_names = "tp"

    state = es.ParameterState.from_layout("out in", mesh_dim_names="tp")
    mesh_state = es.ParameterState.from_layout("out in", mesh=Mesh())
    param = torch.nn.Parameter(torch.ones(2, 3))
    linear = nn.Linear(3, 2, bias=False)

    es.register_parameter_layout(param, "out in", mesh_dim_names="tp")
    es.register_linear_parameters_(linear, mesh_dim_names="tp")

    assert state.shared == ("tp",)
    assert state.tensor_state.replicated_dims == ("tp",)
    assert mesh_state.shared == ("tp",)
    assert es.get_parameter_state(param).shared == ("tp",)
    assert es.get_parameter_state(linear.weight).shared == ("tp",)


def test_parameter_state_from_layout_rejects_non_string_mesh_dim_names():
    for kwargs in (
        {"mesh_dim_names": 0},
        {"mesh_dim_names": (1,)},
        {"mesh_dim_names": torch.tensor([])},
        {"mesh": type("ScalarMesh", (), {"mesh_dim_names": False})()},
        {"mesh": type("Mesh", (), {"mesh_dim_names": (1,)})()},
        {"mesh": type("TensorMesh", (), {"mesh_dim_names": torch.tensor([])})()},
    ):
        try:
            es.ParameterState.from_layout("out in", **kwargs)
        except TypeError as error:
            assert "mesh_dim_names" in str(error)
        else:
            raise AssertionError("Expected non-string mesh dim names to fail")


def test_parameter_state_from_layout_rejects_empty_or_duplicate_mesh_dim_names():
    for kwargs in (
        {"mesh_dim_names": ""},
        {"mesh_dim_names": ("tp", "tp")},
        {"mesh_dim_names": ("dp-dp",)},
        {"mesh_dim_names": ("dp-",)},
        {"mesh_dim_names": ("dp--sp",)},
        {"mesh_dim_names": ("dp", "dp-sp")},
        {"mesh": type("EmptyMesh", (), {"mesh_dim_names": ""})()},
        {"mesh": type("DuplicateMesh", (), {"mesh_dim_names": ("tp", "tp")})()},
        {"mesh": type("RepeatedComponentMesh", (), {"mesh_dim_names": ("dp-dp",)})()},
        {"mesh": type("OverlappingComponentMesh", (), {"mesh_dim_names": ("dp", "dp-sp")})()},
        {"mesh": type("EmptyComponentMesh", (), {"mesh_dim_names": ("dp-",)})()},
    ):
        try:
            es.ParameterState.from_layout("out in", **kwargs)
        except ValueError as error:
            assert "mesh_dim_names" in str(error)
        else:
            raise AssertionError("Expected invalid mesh dim names to fail")


def test_parameter_state_from_layout_rejects_mesh_dim_name_count_mismatch():
    for names in (("tp",), None):
        class Mesh:
            mesh = torch.zeros(1, 1)
            mesh_dim_names = names

        try:
            es.ParameterState.from_layout("out in", mesh=Mesh())
        except ValueError as error:
            assert "mesh_dim_names" in str(error)
        else:
            raise AssertionError("Expected mesh dim name count mismatch to fail")


def test_parameter_state_from_layout_rejects_mesh_dim_names_that_disagree_with_mesh():
    class Mesh:
        mesh = torch.zeros(1, 1)
        mesh_dim_names = ("dp", "sp")

    try:
        es.ParameterState.from_layout("out in", mesh=Mesh(), mesh_dim_names=("foo", "bar"))
    except ValueError as error:
        assert "mesh_dim_names" in str(error)
    else:
        raise AssertionError("Expected mesh_dim_names mismatch to fail")


def test_parameter_state_from_layout_rejects_unknown_mesh_groups():
    cases = (
        {"layout": "out/tp in"},
        {"layout": "out in", "grad": "tp"},
        {"layout": "out in", "grad": "tp:ddp"},
        {"layout": "out in", "init_sync": "tp"},
    )
    for kwargs in cases:
        try:
            es.ParameterState.from_layout(mesh_dim_names=("dp", "sp"), **kwargs)
        except ValueError as error:
            assert "unknown mesh" in str(error)
        else:
            raise AssertionError("Expected unknown mesh group to fail")


def test_parameter_state_from_layout_allows_external_unknown_grad_group():
    state = es.ParameterState.from_layout("out in", mesh_dim_names=("dp", "sp"), grad="tp:external")

    assert state.grad_comm.backend == "external"
    assert state.grad_comm.mesh_dims == ("tp",)


def test_parameter_state_from_layout_allows_known_compound_mesh_groups():
    state = es.ParameterState.from_layout(
        "out in",
        mesh_dim_names=("dp", "sp"),
        grad="dp-sp",
        init_sync="dp-sp",
    )

    assert state.reduce == ("dp-sp",)
    assert state.shared == ("dp-sp",)


def test_parameter_state_from_layout_rejects_unknown_compound_components():
    cases = (
        {"mesh_dim_names": ("dp-sp",), "grad": "dp"},
    )
    for kwargs in cases:
        try:
            es.ParameterState.from_layout("out in", **kwargs)
        except ValueError as error:
            assert "unknown mesh" in str(error)
        else:
            raise AssertionError("Expected unknown compound mesh group to fail")


def test_parameter_state_from_layout_rejects_recombined_compound_candidate_components():
    cases = (
        {"layout": "a/sp1 b/sp2", "mesh_dim_names": ("sp1-sp2",)},
        {"layout": "out/foo-bar", "mesh_dim_names": ("foo-baz", "bar-qux")},
    )
    for kwargs in cases:
        try:
            es.ParameterState.from_layout(**kwargs)
        except ValueError as error:
            assert "unknown mesh" in str(error)
        else:
            raise AssertionError("Expected recombined compound candidate components to fail")


def test_parameter_state_from_layout_rejects_repeated_compound_group_components():
    cases = (
        {"layout": "out/dp-dp in"},
        {"layout": "out in", "grad": "dp-dp"},
        {"layout": "out in", "grad": "dp-dp:external"},
        {"layout": "out in", "init_sync": "dp-dp"},
    )
    for kwargs in cases:
        try:
            es.ParameterState.from_layout(**kwargs)
        except ValueError as error:
            assert "repeated mesh" in str(error)
        else:
            raise AssertionError("Expected repeated compound group component to fail")


def test_parameter_state_from_param_spec_rejects_unknown_mesh_groups():
    cases = (
        es.ParamSpec("out/tp in"),
        es.ParamSpec("out in", shared="sp"),
        es.ParamSpec("out in", reduce="sp"),
    )
    for spec in cases:
        try:
            es.ParameterState.from_param_spec(spec, mesh_dim_names=("dp",))
        except ValueError as error:
            assert "unknown mesh" in str(error)
        else:
            raise AssertionError("Expected unknown ParamSpec mesh group to fail")


def test_register_parameter_layout_revalidates_attached_param_spec_with_mesh_names():
    param = torch.nn.Parameter(torch.ones(2))
    es.set_param_spec(param, es.ParamSpec("o", reduce="foo"))

    try:
        es.register_parameter_layout(param, "o", mesh_dim_names=("dp",))
    except ValueError as error:
        assert "unknown mesh" in str(error)
    else:
        raise AssertionError("Expected attached ParamSpec unknown group to fail")

    state = es.get_parameter_state(param)
    assert state.source == "ParamSpec"
    assert state.reduce == ("foo",)
    assert state.mesh_dim_names == ()


def test_parameter_state_from_layout_rejects_array_like_mesh_dim_names():
    try:
        es.ParameterState.from_layout("out in", mesh_dim_names=torch.tensor([0]))
    except TypeError as error:
        assert "mesh_dim_names" in str(error)
    else:
        raise AssertionError("Expected tensor mesh dim names to fail clearly")


def test_parameter_state_from_spec_accepts_string_mesh_dim_names():
    _, weight, _ = es.parse_sharding("b c, out c [param] -> b out")

    state = es.ParameterState.from_spec(weight, mesh_dim_names="tp")

    assert state.shared == ("tp",)
    assert state.tensor_state.replicated_dims == ("tp",)


def test_einshard_parameter_registration_accepts_string_mesh_dim_names():
    class Mesh:
        mesh_dim_names = "tp"

    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.einshard("b c, c [param] -> b c", x, weight, mesh=Mesh())

    assert es.get_parameter_state(weight).shared == ("tp",)


def test_parameter_state_from_layout_accepts_explicit_metadata():
    state = es.ParameterState.from_layout(
        "out in",
        mesh_dim_names=("dp", "sp"),
        grad="sp:async",
        init_sync="dp",
    )

    assert state.init_sync.mode == "explicit"
    assert state.shared == ("dp",)
    assert state.grad_comm.mode == "explicit"
    assert state.grad_comm.mesh_dims == ("sp",)
    assert state.grad_comm.backend == "native"
    assert state.grad_comm.schedule == "async"


def test_parameter_state_from_layout_rejects_malformed_annotation_values():
    for kwargs in (
        {"grad": ""},
        {"grad": "dp,sp"},
        {"grad": "dp:bogus"},
        {"init_sync": ""},
        {"init_sync": "dp:async"},
    ):
        try:
            es.ParameterState.from_layout("out in", **kwargs)
        except ValueError as error:
            assert "Invalid" in str(error) or "suffix" in str(error)
        else:
            raise AssertionError("Expected malformed layout annotation value to fail")


def test_parameter_state_from_layout_rejects_non_string_annotation_values():
    try:
        es.ParameterState.from_layout("out in", grad=1)
    except TypeError as error:
        assert "string" in str(error)
    else:
        raise AssertionError("Expected non-string grad annotation value to fail")


def test_parameter_state_from_layout_rejects_non_string_layout():
    try:
        es.ParameterState.from_layout(None)
    except TypeError as error:
        assert "layout" in str(error)
    else:
        raise AssertionError("Expected non-string layout to fail")


def test_parameter_state_from_layout_rejects_non_symbolic_layouts():
    for layout in ("out ...", "out out", "(a b)"):
        try:
            es.ParameterState.from_layout(layout)
        except ValueError as error:
            assert "TensorState" in str(error) or "factored axes" in str(error)
        else:
            raise AssertionError("Expected non-symbolic explicit layout to fail")


def test_register_parameter_layout_rejects_non_parameter_tensors():
    tensor = torch.ones(2, 3)

    try:
        es.register_parameter_layout(tensor, "out in")
    except TypeError as error:
        assert "torch.nn.Parameter" in str(error)
    else:
        raise AssertionError("Expected non-Parameter layout registration to fail")


def test_register_parameter_layout_validates_parameter_rank():
    param = torch.nn.Parameter(torch.ones(2, 3))

    try:
        es.register_parameter_layout(param, "out")
    except ValueError as error:
        assert "rank" in str(error)
    else:
        raise AssertionError("Expected rank mismatch to fail")

    assert es.get_parameter_state(param) is None


def test_register_parameter_layout_rejects_ellipsis_layout():
    param = torch.nn.Parameter(torch.ones(2, 3))

    try:
        es.register_parameter_layout(param, "out ...")
    except ValueError as error:
        assert "TensorState" in str(error)
    else:
        raise AssertionError("Expected ellipsis layout registration to fail")

    assert es.get_parameter_state(param) is None


def test_register_parameter_layout_merges_with_later_formula_grad(dist_env, mesh_2d):
    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    es.register_parameter_layout(weight, "c", mesh=mesh_2d)
    state = es.get_parameter_state(weight)
    assert state.source == "layout"
    assert state.shared == ("dp", "sp")
    assert state.grad_comm.mode == "none"

    es.einshard(
        "b/dp c, c [param, grad=async] -> b/dp c",
        x,
        weight,
        mesh=mesh_2d,
    )

    state = es.get_parameter_state(weight)
    assert state.source == "layout"
    assert state.shared == ("dp", "sp")
    assert state.grad_comm.mesh_dims == ("dp",)
    assert state.grad_comm.schedule == "async"


def test_register_parameter_layout_source_reflects_param_spec_merge():
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_param_spec(weight, es.ParamSpec("c"))

    es.register_parameter_layout(weight, "c", grad="dp")

    state = es.get_parameter_state(weight)
    assert state.source == "ParamSpec+layout"
    assert state.reduce == ("dp",)


def test_register_linear_parameters_registers_weight_and_bias():
    module = nn.Linear(3, 2)

    assert es.register_linear_parameters_(
        module,
        weight_layout="out/tp in",
        mesh_dim_names=("tp", "sp"),
        weight_grad="sp",
        bias_grad="sp",
    ) is module

    weight_state = es.get_parameter_state(module.weight)
    bias_state = es.get_parameter_state(module.bias)
    assert weight_state.source == "linear"
    assert weight_state.layout_shard_dims == ("tp",)
    assert weight_state.shared == ("sp",)
    assert weight_state.reduce == ("sp",)
    assert bias_state.source == "linear"
    assert bias_state.layout_shard_dims == ("tp",)
    assert bias_state.shared == ("sp",)
    assert bias_state.reduce == ("sp",)
    assert repr(bias_state.spec.axes) == "out / tp"


def test_register_linear_parameters_supports_modules_without_bias():
    module = nn.Linear(3, 2, bias=False)

    es.register_linear_parameters_(module, weight_layout="out in", weight_grad="dp")

    assert es.get_parameter_state(module.weight).reduce == ("dp",)


def test_register_linear_parameters_rejects_bias_metadata_without_bias():
    module = nn.Linear(3, 2, bias=False)

    try:
        es.register_linear_parameters_(module, weight_layout="out in", bias_grad="dp")
    except ValueError as error:
        assert "bias" in str(error)
    else:
        raise AssertionError("Expected bias metadata on a biasless module to fail")

    assert es.get_parameter_state(module.weight) is None


def test_register_linear_parameters_rejects_explicit_bias_layout_mismatch():
    module = nn.Linear(3, 2)

    try:
        es.register_linear_parameters_(module, weight_layout="out/tp in", bias_layout="in")
    except ValueError as error:
        assert "bias layout" in str(error)
    else:
        raise AssertionError("Expected mismatched linear bias layout to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_linear_parameters_rejects_bias_shape_mismatch():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2, 3))
    module.bias = torch.nn.Parameter(torch.ones(3))

    try:
        es.register_linear_parameters_(module)
    except ValueError as error:
        assert "bias shape" in str(error)
    else:
        raise AssertionError("Expected linear bias shape mismatch to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_linear_parameters_rejects_non_linear_weight_rank():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2))
    module.bias = None

    try:
        es.register_linear_parameters_(module, weight_layout="out")
    except ValueError as error:
        assert "weight" in str(error)
        assert "rank 2" in str(error)
    else:
        raise AssertionError("Expected non-2D linear weight to fail")

    assert es.get_parameter_state(module.weight) is None


def test_register_linear_parameters_rejects_non_linear_bias_rank():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2, 3))
    module.bias = torch.nn.Parameter(torch.ones(2, 1))

    try:
        es.register_linear_parameters_(module)
    except ValueError as error:
        assert "bias" in str(error)
        assert "rank 1" in str(error)
    else:
        raise AssertionError("Expected non-1D linear bias to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_linear_parameters_is_atomic_on_metadata_conflict():
    module = nn.Linear(3, 2)
    es.set_param_spec(module.bias, es.ParamSpec("other"))

    try:
        es.register_linear_parameters_(module, weight_layout="out in")
    except ValueError as error:
        assert "different layout" in str(error)
    else:
        raise AssertionError("Expected bias metadata conflict to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias).source == "ParamSpec"


def test_register_linear_parameters_rejects_aliased_parameters_atomically():
    module = nn.Module()
    shared = torch.nn.Parameter(torch.ones(2, 3))
    module.weight = shared
    module.bias = shared

    try:
        es.register_linear_parameters_(
            module,
            weight_layout="out in",
            bias_layout="out",
            weight_grad="dp",
            bias_grad="sp",
        )
    except ValueError as error:
        assert "bias" in str(error)
    else:
        raise AssertionError("Expected aliased linear parameters to fail")

    assert es.get_parameter_state(shared) is None


def test_register_linear_parameters_rejects_missing_weight_parameter():
    module = nn.Module()

    try:
        es.register_linear_parameters_(module)
    except ValueError as error:
        assert "weight" in str(error)
    else:
        raise AssertionError("Expected missing weight parameter to fail")


def test_register_linear_parameters_rejects_non_string_layouts():
    module = nn.Linear(3, 2)

    try:
        es.register_linear_parameters_(module, weight_layout=None)
    except TypeError as error:
        assert "layout" in str(error)
    else:
        raise AssertionError("Expected non-string weight layout to fail")


def test_register_conv_parameters_registers_weight_and_bias():
    module = nn.Conv2d(3, 2, kernel_size=3)

    es.register_conv_parameters_(module, mesh_dim_names=("tp", "sp"), grad="sp")

    weight_state = es.get_parameter_state(module.weight)
    bias_state = es.get_parameter_state(module.bias)
    assert weight_state.source == "conv"
    assert bias_state.source == "conv"
    assert repr(weight_state.spec.axes) == "out × in × kh × kw"
    assert repr(bias_state.spec.axes) == "out"
    assert weight_state.shared == ("tp", "sp")
    assert bias_state.shared == ("tp", "sp")
    assert weight_state.reduce == ("sp",)
    assert bias_state.reduce == ("sp",)


def test_register_conv_parameters_derives_rank_specific_weight_layouts():
    conv1 = nn.Conv1d(3, 2, kernel_size=3, bias=False)
    conv3 = nn.Conv3d(3, 2, kernel_size=3, bias=False)

    es.register_conv_parameters_(conv1)
    es.register_conv_parameters_(conv3)

    assert repr(es.get_parameter_state(conv1.weight).spec.axes) == "out × in × k"
    assert repr(es.get_parameter_state(conv3.weight).spec.axes) == "out × in × kd × kh × kw"


def test_register_conv_parameters_derives_sharded_bias_layout():
    module = nn.Conv2d(3, 2, kernel_size=3)

    es.register_conv_parameters_(
        module,
        weight_layout="out/tp in kh kw",
        mesh_dim_names=("tp", "sp"),
        weight_grad="sp",
        bias_grad="sp",
    )

    assert repr(es.get_parameter_state(module.weight).spec.axes) == "out / tp × in × kh × kw"
    assert repr(es.get_parameter_state(module.bias).spec.axes) == "out / tp"
    assert es.get_parameter_state(module.weight).shared == ("sp",)
    assert es.get_parameter_state(module.bias).shared == ("sp",)


def test_register_conv_parameters_supports_biasless_modules():
    module = nn.Conv2d(3, 2, kernel_size=3, bias=False)

    es.register_conv_parameters_(module, grad="dp")

    assert es.get_parameter_state(module.weight).reduce == ("dp",)


def test_register_conv_parameters_rejects_bias_metadata_without_bias():
    module = nn.Conv2d(3, 2, kernel_size=3, bias=False)

    try:
        es.register_conv_parameters_(module, bias_grad="dp")
    except ValueError as error:
        assert "bias" in str(error)
    else:
        raise AssertionError("Expected bias metadata on a biasless conv module to fail")

    assert es.get_parameter_state(module.weight) is None


def test_register_conv_parameters_rejects_explicit_bias_layout_mismatch():
    module = nn.Conv2d(3, 2, kernel_size=3)

    try:
        es.register_conv_parameters_(module, bias_layout="in")
    except ValueError as error:
        assert "bias" in str(error)
        assert "layout" in str(error)
    else:
        raise AssertionError("Expected mismatched conv bias layout to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_conv_parameters_rejects_grouped_convolutions():
    module = nn.Conv2d(4, 4, kernel_size=3, groups=2)

    try:
        es.register_conv_parameters_(module)
    except NotImplementedError as error:
        assert "groups" in str(error)
    else:
        raise AssertionError("Expected grouped conv registration to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_conv_parameters_rejects_conv_transpose():
    module = nn.ConvTranspose2d(3, 2, kernel_size=3)

    try:
        es.register_conv_parameters_(module)
    except NotImplementedError as error:
        assert "ConvTranspose" in str(error)
    else:
        raise AssertionError("Expected ConvTranspose registration to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_conv_parameters_rejects_conv_transpose_subclasses():
    class CustomTranspose(nn.ConvTranspose2d):
        pass

    module = CustomTranspose(3, 3, kernel_size=3)

    try:
        es.register_conv_parameters_(module)
    except NotImplementedError as error:
        assert "ConvTranspose" in str(error)
    else:
        raise AssertionError("Expected ConvTranspose subclass registration to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_conv_parameters_rejects_bias_shape_mismatch():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(2, 3, 3, 3))
    module.bias = torch.nn.Parameter(torch.ones(4))

    try:
        es.register_conv_parameters_(module)
    except ValueError as error:
        assert "bias shape" in str(error)
    else:
        raise AssertionError("Expected conv bias shape mismatch to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_conv_parameters_is_atomic_on_rank_mismatch():
    module = nn.Conv2d(3, 2, kernel_size=3)

    try:
        es.register_conv_parameters_(module, weight_layout="out in")
    except ValueError as error:
        assert "rank" in str(error)
    else:
        raise AssertionError("Expected conv layout rank mismatch to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_norm_parameters_registers_weight_and_bias():
    module = nn.LayerNorm(3)

    es.register_norm_parameters_(module, layout="c", mesh_dim_names=("dp", "sp"), grad="dp-sp")

    weight_state = es.get_parameter_state(module.weight)
    bias_state = es.get_parameter_state(module.bias)
    assert weight_state.source == "norm"
    assert bias_state.source == "norm"
    assert repr(weight_state.spec.axes) == "c"
    assert repr(bias_state.spec.axes) == "c"
    assert weight_state.shared == ("dp", "sp")
    assert bias_state.shared == ("dp", "sp")
    assert weight_state.reduce == ("dp-sp",)
    assert bias_state.reduce == ("dp-sp",)


def test_register_norm_parameters_supports_weight_only_modules():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    module.bias = None

    es.register_norm_parameters_(module, layout="c", grad="dp")

    assert es.get_parameter_state(module.weight).reduce == ("dp",)


def test_register_norm_parameters_uses_grad_overrides():
    module = nn.LayerNorm(3)

    es.register_norm_parameters_(module, layout="c", grad="dp", bias_grad="sp")

    assert es.get_parameter_state(module.weight).reduce == ("dp",)
    assert es.get_parameter_state(module.bias).reduce == ("sp",)


def test_register_norm_parameters_accepts_weight_layout_alias():
    module = nn.LayerNorm(3)

    es.register_norm_parameters_(module, weight_layout="hidden", grad="dp")

    assert repr(es.get_parameter_state(module.weight).spec.axes) == "hidden"
    assert repr(es.get_parameter_state(module.bias).spec.axes) == "hidden"


def test_register_norm_parameters_rejects_conflicting_layout_aliases():
    module = nn.LayerNorm(3)

    try:
        es.register_norm_parameters_(module, layout="c", weight_layout="hidden")
    except ValueError as error:
        assert "weight" in str(error)
        assert "layout" in str(error)
    else:
        raise AssertionError("Expected conflicting norm layout aliases to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_norm_parameters_allows_matching_layout_aliases():
    module = nn.LayerNorm(3)

    es.register_norm_parameters_(module, layout="hidden", weight_layout="hidden", grad="dp")

    assert repr(es.get_parameter_state(module.weight).spec.axes) == "hidden"
    assert repr(es.get_parameter_state(module.bias).spec.axes) == "hidden"


def test_register_norm_parameters_rejects_explicit_none_layout():
    module = nn.LayerNorm(3)

    try:
        es.register_norm_parameters_(module, layout=None, weight_layout="hidden")
    except TypeError as error:
        assert "layout" in str(error)
    else:
        raise AssertionError("Expected explicit None norm layout to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_norm_parameters_rejects_bias_metadata_without_bias():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    module.bias = None

    try:
        es.register_norm_parameters_(module, layout="c", bias_grad="dp")
    except ValueError as error:
        assert "bias" in str(error)
    else:
        raise AssertionError("Expected bias metadata on a biasless norm module to fail")

    assert es.get_parameter_state(module.weight) is None


def test_register_norm_parameters_rejects_bias_layout_mismatch():
    module = nn.LayerNorm(3)

    try:
        es.register_norm_parameters_(module, layout="c", bias_layout="d")
    except ValueError as error:
        assert "bias" in str(error)
        assert "layout" in str(error)
    else:
        raise AssertionError("Expected mismatched norm bias layout to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_norm_parameters_rejects_bias_shape_mismatch():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    module.bias = torch.nn.Parameter(torch.ones(4))

    try:
        es.register_norm_parameters_(module, layout="c")
    except ValueError as error:
        assert "bias shape" in str(error)
    else:
        raise AssertionError("Expected norm bias shape mismatch to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_norm_parameters_is_atomic_on_rank_mismatch():
    module = nn.LayerNorm(3)

    try:
        es.register_norm_parameters_(module, layout="h w")
    except ValueError as error:
        assert "rank" in str(error)
    else:
        raise AssertionError("Expected norm layout rank mismatch to fail")

    assert es.get_parameter_state(module.weight) is None
    assert es.get_parameter_state(module.bias) is None


def test_register_norm_parameters_rejects_missing_weight_parameter():
    module = nn.Module()
    module.bias = torch.nn.Parameter(torch.ones(3))

    try:
        es.register_norm_parameters_(module)
    except ValueError as error:
        assert "weight" in str(error)
    else:
        raise AssertionError("Expected missing norm weight parameter to fail")


def test_register_module_parameter_layouts_registers_named_parameters():
    module = nn.Module()
    module.qkv_weight = torch.nn.Parameter(torch.ones(6, 3))
    module.qkv_bias = torch.nn.Parameter(torch.ones(6))

    es.register_module_parameter_layouts_(
        module,
        {"qkv_weight": "out/tp in", "qkv_bias": "out/tp"},
        mesh_dim_names=("tp", "sp"),
        grad={"qkv_weight": "sp", "qkv_bias": "sp"},
        init_sync="sp",
    )

    weight_state = es.get_parameter_state(module.qkv_weight)
    bias_state = es.get_parameter_state(module.qkv_bias)
    assert weight_state.source == "module"
    assert bias_state.source == "module"
    assert weight_state.shared == ("sp",)
    assert bias_state.shared == ("sp",)
    assert weight_state.reduce == ("sp",)
    assert bias_state.reduce == ("sp",)
    assert repr(weight_state.spec.axes) == "out / tp × in"
    assert repr(bias_state.spec.axes) == "out / tp"


def test_register_module_parameter_layouts_supports_nested_names():
    module = nn.Module()
    module.proj = nn.Linear(3, 2)

    es.register_module_parameter_layouts_(module, {"proj.weight": "out in"}, grad="dp")

    state = es.get_parameter_state(module.proj.weight)
    assert state.source == "module"
    assert state.reduce == ("dp",)


def test_register_module_parameter_layouts_rejects_unknown_metadata_keys():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))

    try:
        es.register_module_parameter_layouts_(
            module,
            {"weight": "c"},
            grad={"weight": "dp", "missing": "sp"},
        )
    except ValueError as error:
        assert "unknown parameter names" in str(error)
        assert "missing" in str(error)
    else:
        raise AssertionError("Expected unknown grad metadata key to fail")

    assert es.get_parameter_state(module.weight) is None


def test_register_module_parameter_layouts_is_atomic_on_missing_parameter():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))

    try:
        es.register_module_parameter_layouts_(module, {"weight": "c", "missing": "c"})
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("Expected missing named parameter to fail")

    assert es.get_parameter_state(module.weight) is None


def test_register_module_parameter_layouts_merges_aliased_parameters_atomically():
    module = nn.Module()
    shared = torch.nn.Parameter(torch.ones(3))
    module.weight = shared
    module.bias = shared

    try:
        es.register_module_parameter_layouts_(module, {"weight": "c", "bias": "d"})
    except ValueError as error:
        assert "different layout" in str(error)
    else:
        raise AssertionError("Expected aliased parameter layout conflict to fail")

    assert es.get_parameter_state(shared) is None


def test_register_module_parameter_layouts_rejects_empty_or_non_mapping_layouts():
    module = nn.Module()

    try:
        es.register_module_parameter_layouts_(module, [])
    except TypeError as error:
        assert "mapping" in str(error)
    else:
        raise AssertionError("Expected non-mapping layouts to fail")

    try:
        es.register_module_parameter_layouts_(module, {})
    except ValueError as error:
        assert "must not be empty" in str(error)
    else:
        raise AssertionError("Expected empty layouts mapping to fail")


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


def test_validate_module_parameter_states_accepts_valid_states():
    module = nn.Linear(3, 2)
    es.register_linear_parameters_(module, weight_grad="dp", bias_grad="dp")

    assert es.validate_module_parameter_states_(module) is module


def test_validate_module_parameter_states_rejects_pending_native_grad():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(
        module.weight,
        es.ParameterState.from_layout("c", grad="async"),
    )

    try:
        es.validate_module_parameter_states_(module)
    except ValueError as error:
        assert "weight" in str(error)
        assert "pending inference" in str(error)
    else:
        raise AssertionError("Expected pending native grad metadata to fail validation")

    assert es.validate_module_parameter_states_(module, allow_pending=True) is module


def test_validate_module_parameter_states_rejects_pending_ddp_grad():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(
        module.weight,
        es.ParameterState.from_layout("c", grad="ddp"),
    )

    try:
        es.validate_module_parameter_states_(module)
    except ValueError as error:
        assert "weight" in str(error)
        assert "pending inference" in str(error)
    else:
        raise AssertionError("Expected pending DDP grad metadata to fail validation")

    assert es.validate_module_parameter_states_(module, allow_pending=True) is module


def test_finalize_parameter_grad_comm_resolves_pending_native_grad():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(module.weight, es.ParameterState.from_layout("c", grad="async"))

    assert es.finalize_parameter_grad_comm_(module.weight, "dp", mesh_dim_names=("dp", "sp")) is module.weight

    state = es.get_parameter_state(module.weight)
    assert state.reduce == ("dp",)
    assert state.grad_comm.backend == "native"
    assert state.grad_comm.schedule == "async"
    assert state.mesh_dim_names == ("dp", "sp")
    assert es.validate_module_parameter_states_(module) is module


def test_finalize_module_parameter_grad_comm_resolves_pending_ddp_grad():
    module = nn.Linear(3, 2, bias=False)
    es.set_parameter_state(module.weight, es.ParameterState.from_layout("o c", grad="ddp"))

    assert es.finalize_module_parameter_grad_comm_(module, {"weight": "dp:ddp"}) is module

    state = es.get_parameter_state(module.weight)
    assert state.grad_comm.backend == "ddp"
    assert state.grad_comm.mesh_dims == ("dp",)
    assert es.validate_module_parameter_states_(module) is module


def test_finalize_module_parameter_grad_comm_resolves_pending_external_grad():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(module.weight, es.ParameterState.from_layout("c", grad="external"))

    es.finalize_module_parameter_grad_comm_(module, {"weight": "dp:external"})

    state = es.get_parameter_state(module.weight)
    assert state.grad_comm.backend == "external"
    assert state.grad_comm.mesh_dims == ("dp",)
    assert state.reduce == ()


def test_finalize_parameter_grad_comm_rejects_inferred_policy_values():
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(weight, es.ParameterState.from_layout("c", grad="async"))

    for value in ("async", "ddp", "external"):
        try:
            es.finalize_parameter_grad_comm_(weight, value)
        except ValueError as error:
            assert "explicit mesh group" in str(error)
        else:
            raise AssertionError(f"Expected {value!r} to fail grad finalization")

    assert es.get_parameter_state(weight).grad_comm.pending_inference


def test_finalize_parameter_grad_comm_rejects_nonpending_grad():
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(weight, es.ParameterState.from_layout("c", grad="dp"))

    try:
        es.finalize_parameter_grad_comm_(weight, "dp")
    except ValueError as error:
        assert "not pending" in str(error)
    else:
        raise AssertionError("Expected non-pending grad finalization to fail")


def test_finalize_parameter_grad_comm_rejects_layout_shard_overlap():
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(weight, es.ParameterState.from_layout("c/tp", grad="async"))

    try:
        es.finalize_parameter_grad_comm_(weight, "tp")
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected finalization over a sharded layout dim to fail")

    assert es.get_parameter_state(weight).grad_comm.pending_inference


def test_finalize_parameter_grad_comm_validates_supplied_mesh_groups(mesh_2d):
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(weight, es.ParameterState.from_layout("c", grad="async"))

    try:
        es.finalize_parameter_grad_comm_(weight, "dp-sp", mesh=mesh_2d)
    except ValueError as error:
        assert "wrap_mesh" in str(error)
    else:
        raise AssertionError("Expected unwrapped compound finalized grad group to fail")

    assert es.get_parameter_state(weight).grad_comm.pending_inference
    es.finalize_parameter_grad_comm_(weight, "dp-sp", mesh=es.wrap_mesh(mesh_2d))
    assert es.get_parameter_state(weight).reduce == ("dp-sp",)


def test_finalize_parameter_grad_comm_skips_external_mesh_group_validation(mesh_2d):
    weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(weight, es.ParameterState.from_layout("c", grad="external"))

    es.finalize_parameter_grad_comm_(weight, "missing:external", mesh=mesh_2d)

    state = es.get_parameter_state(weight)
    assert state.grad_comm.backend == "external"
    assert state.grad_comm.mesh_dims == ("missing",)


def test_finalize_module_parameter_grad_comm_is_atomic_on_failure():
    module = nn.Linear(3, 2)
    es.set_parameter_state(module.weight, es.ParameterState.from_layout("o c", grad="async"))
    es.set_parameter_state(module.bias, es.ParameterState.from_layout("o", grad="async"))

    try:
        es.finalize_module_parameter_grad_comm_(
            module,
            {"weight": "dp", "bias": "missing"},
            mesh_dim_names=("dp",),
        )
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("Expected atomic grad finalization failure")

    assert es.get_parameter_state(module.weight).grad_comm.pending_inference
    assert es.get_parameter_state(module.bias).grad_comm.pending_inference


def test_validate_module_parameter_states_checks_mesh_groups(mesh_2d):
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(
        module.weight,
        es.ParameterState.from_layout("c", grad="missing"),
    )

    try:
        es.validate_module_parameter_states_(module, mesh_2d)
    except ValueError as error:
        assert "weight" in str(error)
        assert "missing" in str(error)
    else:
        raise AssertionError("Expected unknown mesh group to fail validation")


def test_validate_module_parameter_states_checks_layout_mesh_groups(mesh_2d):
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(
        module.weight,
        es.ParameterState.from_layout("c/missing"),
    )

    try:
        es.validate_module_parameter_states_(module, mesh_2d)
    except ValueError as error:
        assert "weight" in str(error)
        assert "missing" in str(error)
    else:
        raise AssertionError("Expected unknown layout mesh group to fail validation")


def test_validate_module_parameter_states_requires_wrapped_compound_layout_groups(mesh_2d):
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(
        module.weight,
        es.ParameterState.from_layout("c/dp-sp"),
    )

    try:
        es.validate_module_parameter_states_(module, mesh_2d)
    except ValueError as error:
        assert "weight" in str(error)
        assert "wrap_mesh" in str(error)
    else:
        raise AssertionError("Expected unwrapped compound layout group to fail validation")

    assert es.validate_module_parameter_states_(module, es.wrap_mesh(mesh_2d)) is module


def test_validate_module_parameter_states_detects_raw_rank_mismatch():
    module = nn.Linear(3, 2)
    es.set_parameter_state(module.weight, es.ParameterState.from_layout("c"))

    try:
        es.validate_module_parameter_states_(module)
    except ValueError as error:
        assert "weight" in str(error)
        assert "rank" in str(error)
    else:
        raise AssertionError("Expected rank-mismatched raw state to fail validation")


def test_validate_module_parameter_states_rejects_non_state_metadata():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(module.weight, object())

    try:
        es.validate_module_parameter_states_(module)
    except ValueError as error:
        assert "weight" in str(error)
        assert "ParameterState" in str(error)
    else:
        raise AssertionError("Expected non-ParameterState metadata to fail validation")


def test_validate_module_parameter_states_rejects_stale_layout_shard_dims():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    state = replace(es.ParameterState.from_layout("c"), layout_shard_dims=("missing",))
    es.set_parameter_state(module.weight, state)

    try:
        es.validate_module_parameter_states_(module)
    except ValueError as error:
        assert "weight" in str(error)
        assert "layout_shard_dims" in str(error)
    else:
        raise AssertionError("Expected stale cached layout shard dims to fail validation")


def test_validate_module_parameter_states_rejects_malformed_init_sync():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    state = replace(
        es.ParameterState.from_layout("c"),
        init_sync=es.ParameterInitSync(mode="bogus", mesh_dims=("dp",)),
    )
    es.set_parameter_state(module.weight, state)

    try:
        es.validate_module_parameter_states_(module)
    except ValueError as error:
        assert "weight" in str(error)
        assert "ParameterInitSync" in str(error)
    else:
        raise AssertionError("Expected malformed init-sync metadata to fail validation")


def test_validate_module_parameter_states_rejects_inferred_init_sync_overlap():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    state = replace(
        es.ParameterState.from_layout("c/tp", mesh_dim_names=("tp", "sp")),
        init_sync=es.ParameterInitSync(mode="inferred", mesh_dims=("tp",)),
    )
    es.set_parameter_state(module.weight, state)

    try:
        es.validate_module_parameter_states_(module)
    except ValueError as error:
        assert "weight" in str(error)
        assert "init_sync" in str(error)
        assert "tp" in str(error)
    else:
        raise AssertionError("Expected inferred init-sync layout overlap to fail validation")


def test_validate_module_parameter_states_rejects_unknown_inferred_init_sync():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    state = replace(
        es.ParameterState.from_layout("c", mesh_dim_names=("dp",)),
        init_sync=es.ParameterInitSync(mode="inferred", mesh_dims=("missing",)),
    )
    es.set_parameter_state(module.weight, state)

    try:
        es.validate_module_parameter_states_(module)
    except ValueError as error:
        assert "weight" in str(error)
        assert "missing" in str(error)
    else:
        raise AssertionError("Expected unknown inferred init-sync group to fail validation")


def test_validate_module_parameter_states_rejects_incomplete_inferred_init_sync():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    state = replace(
        es.ParameterState.from_layout("c", mesh_dim_names=("dp", "sp")),
        init_sync=es.ParameterInitSync(mode="inferred", mesh_dims=("dp",)),
    )
    es.set_parameter_state(module.weight, state)

    try:
        es.validate_module_parameter_states_(module)
    except ValueError as error:
        assert "weight" in str(error)
        assert "inferred init_sync" in str(error)
    else:
        raise AssertionError("Expected incomplete inferred init-sync metadata to fail validation")


def test_validate_module_parameter_states_allows_explicit_init_sync_subset():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    es.set_parameter_state(
        module.weight,
        es.ParameterState.from_layout("c", mesh_dim_names=("dp", "sp"), init_sync="dp"),
    )

    assert es.validate_module_parameter_states_(module) is module


def test_validate_module_parameter_states_rejects_malformed_grad_comm():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    state = replace(
        es.ParameterState.from_layout("c"),
        grad_comm=es.ParameterGradComm(mode="explicit", mesh_dims=("dp",), backend="bogus"),
    )
    es.set_parameter_state(module.weight, state)

    try:
        es.validate_module_parameter_states_(module)
    except ValueError as error:
        assert "weight" in str(error)
        assert "ParameterGradComm" in str(error)
    else:
        raise AssertionError("Expected malformed grad metadata to fail validation")


def test_validate_module_parameter_states_rejects_partial_specs():
    module = nn.Module()
    module.weight = torch.nn.Parameter(torch.ones(3))
    spec = es.parse_sharding("c // dp -> c // dp")[0]
    es.set_parameter_state(module.weight, es.ParameterState(spec=spec))

    try:
        es.validate_module_parameter_states_(module)
    except ValueError as error:
        assert "weight" in str(error)
        assert "axis layout" in str(error)
    else:
        raise AssertionError("Expected partial-bearing parameter state to fail validation")


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


def test_einshard_rejects_unknown_inferred_parameter_grad_mesh_dim():
    class Mesh:
        mesh_dim_names = ("dp",)

    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    try:
        es.einshard("b/foo c, c [param, grad=async] -> b/foo c", x, weight, mesh=Mesh())
    except ValueError as error:
        assert "unknown mesh" in str(error)
    else:
        raise AssertionError("Expected unknown inferred grad mesh dim to fail")

    assert es.get_parameter_state(weight) is None


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


def test_parameter_operand_state_rejects_overlapping_inferred_grad_groups():
    weight = torch.nn.Parameter(torch.ones(3))
    input_spec, param_spec, output_spec = es.parse_sharding(
        "b/dp-sp c, c [param, grad=async] -> b/sp-dp c"
    )

    try:
        parameter_operand_state(
            weight,
            (input_spec, param_spec),
            output_spec,
            1,
            mesh_dim_names=("dp", "sp"),
        )
    except ValueError as error:
        assert "grad" in str(error)
        assert "overlap" in str(error)
    else:
        raise AssertionError("Expected overlapping inferred gradient groups to fail")


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


def test_einshard_ignores_bad_mesh_dim_names_without_parameter_annotations():
    class BadMesh:
        mesh_dim_names = (1,)

    x = torch.ones(2, 3)

    assert_close(es.einshard("b c -> b c", x, mesh=BadMesh()), x)


def test_einshard_validates_bad_mesh_dim_names_with_parameter_annotations():
    class BadMesh:
        mesh_dim_names = (1,)

    x = torch.ones(2, 3)
    weight = torch.nn.Parameter(torch.ones(3))

    try:
        es.einshard("b c, c [param] -> b c", x, weight, mesh=BadMesh())
    except TypeError as error:
        assert "mesh_dim_names" in str(error)
    else:
        raise AssertionError("Expected invalid parameter metadata mesh names to fail")


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


def test_native_grad_reduction_hooks_allreduce_concrete_native_state(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    state = es.ParameterState.from_spec(
        es.parse_sharding("o c [param, grad=dp-sp:async] -> o c")[0]
    )
    es.set_parameter_state(model.weight, state)
    handle = es.register_native_grad_reduction_hooks_(model, mesh)

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    model(x).sum().backward()
    handle.wait()

    world_size = dist.get_world_size()
    expected = float(world_size * (world_size + 1) // 2)
    assert_close(model.weight.grad, torch.full_like(model.weight.grad, expected))
    assert handle.pending == 0
    handle.remove()


def test_native_grad_reduction_hooks_support_param_specs(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    es.set_param_spec(model.weight, es.ParamSpec("o c", reduce="dp-sp"))
    handle = es.register_native_grad_reduction_hooks_(model, mesh)

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    model(x).sum().backward()

    world_size = dist.get_world_size()
    expected = float(world_size * (world_size + 1) // 2)
    assert_close(model.weight.grad, torch.full_like(model.weight.grad, expected))
    assert handle.pending == 0
    handle.remove()


def test_native_grad_reduction_hooks_skip_external_and_ddp_backends(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    for annotation in ("grad=dp-sp:external", "grad=external", "grad=dp-sp:ddp", "grad=ddp"):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1.0)
        state = es.ParameterState.from_spec(
            es.parse_sharding(f"o c [param, {annotation}] -> o c")[0]
        )
        es.set_parameter_state(model.weight, state)
        handle = es.register_native_grad_reduction_hooks_(model, mesh)

        x = torch.tensor([[float(dist.get_rank() + 1)]])
        model(x).sum().backward()
        handle.wait()

        assert_close(model.weight.grad, torch.tensor([[float(dist.get_rank() + 1)]]))
        handle.remove()


def test_native_grad_reduction_hooks_support_gradient_accumulation(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    state = es.ParameterState.from_spec(
        es.parse_sharding("o c [param, grad=dp-sp] -> o c")[0]
    )
    es.set_parameter_state(model.weight, state)
    handle = es.register_native_grad_reduction_hooks_(model, mesh)

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    model(x).sum().backward()
    model(x).sum().backward()

    world_size = dist.get_world_size()
    expected = 2.0 * float(world_size * (world_size + 1) // 2)
    assert_close(model.weight.grad, torch.full_like(model.weight.grad, expected))
    handle.remove()


def test_native_grad_reduction_hooks_reject_pending_native_grad(mesh_2d):
    model = nn.Linear(1, 1, bias=False)
    state = es.ParameterState.from_spec(
        es.parse_sharding("o c [param, grad=async] -> o c")[0]
    )
    es.set_parameter_state(model.weight, state)

    try:
        es.register_native_grad_reduction_hooks_(model, mesh_2d)
    except ValueError as error:
        assert "pending inference" in str(error)
    else:
        raise AssertionError("Expected pending gradient communication to fail")


def test_native_grad_reduction_hooks_validate_before_registering(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    model = nn.Module()
    model.left = nn.Linear(1, 1, bias=False)
    model.right = nn.Linear(1, 1, bias=False)
    es.set_parameter_state(
        model.left.weight,
        es.ParameterState.from_spec(es.parse_sharding("o c [param, grad=dp-sp] -> o c")[0]),
    )
    es.set_parameter_state(
        model.right.weight,
        es.ParameterState.from_spec(es.parse_sharding("o c [param, grad=async] -> o c")[0]),
    )

    try:
        es.register_native_grad_reduction_hooks_(model, mesh)
    except ValueError as error:
        assert "pending inference" in str(error)
    else:
        raise AssertionError("Expected pending gradient communication to fail")

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    model.left(x).sum().backward()

    assert_close(model.left.weight.grad, torch.tensor([[float(dist.get_rank() + 1)]]))


def test_native_grad_reduction_hooks_failure_does_not_lazily_attach_legacy_state(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    model = nn.Module()
    model.left = nn.Linear(1, 1, bias=False)
    model.right = nn.Linear(1, 1, bias=False)
    es.set_parameter_state(
        model.left.weight,
        es.ParameterState.from_spec(es.parse_sharding("o c [param, grad=dp-sp] -> o c")[0]),
    )
    setattr(model.right.weight, PARAM_SPEC_ATTR, es.ParamSpec("o c", reduce="dp-sp"))
    model.right.weight.requires_grad_(False)

    try:
        es.register_native_grad_reduction_hooks_(model, mesh)
    except ValueError as error:
        assert "does not require gradients" in str(error)
    else:
        raise AssertionError("Expected frozen parameter hook registration to fail")

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    model.left(x).sum().backward()

    assert_close(model.left.weight.grad, torch.tensor([[float(dist.get_rank() + 1)]]))
    assert getattr(model.right.weight, PARAM_STATE_ATTR, None) is None


def test_native_grad_reduction_hooks_validate_requires_grad_before_registering(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    model = nn.Module()
    model.left = nn.Linear(1, 1, bias=False)
    model.right = nn.Linear(1, 1, bias=False)
    model.right.weight.requires_grad_(False)
    state = es.ParameterState.from_spec(es.parse_sharding("o c [param, grad=dp-sp] -> o c")[0])
    es.set_parameter_state(model.left.weight, state)
    es.set_parameter_state(model.right.weight, state)

    try:
        es.register_native_grad_reduction_hooks_(model, mesh)
    except ValueError as error:
        assert "does not require gradients" in str(error)
    else:
        raise AssertionError("Expected frozen parameter hook registration to fail")

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    model.left(x).sum().backward()

    assert_close(model.left.weight.grad, torch.tensor([[float(dist.get_rank() + 1)]]))


def test_native_grad_reduction_hook_remove_detaches_hooks(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    state = es.ParameterState.from_spec(
        es.parse_sharding("o c [param, grad=dp-sp] -> o c")[0]
    )
    es.set_parameter_state(model.weight, state)
    handle = es.register_native_grad_reduction_hooks_(model, mesh)
    handle.remove()

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    model(x).sum().backward()

    assert_close(model.weight.grad, torch.tensor([[float(dist.get_rank() + 1)]]))


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

    try:
        es.register_grad_reduction_hook_(ddp, mesh_2d, combined_reduce="sp")
    except ValueError as error:
        assert "combined_reduce_group" in str(error)
    else:
        raise AssertionError("Expected missing combined_reduce_group to fail")


def test_ddp_grad_reduction_hook_validates_parameter_reduce_groups(dist_env, mesh_2d):
    model = nn.Linear(1, 1, bias=False)
    es.set_param_spec(model.weight, es.ParamSpec("o c", reduce="foo"))
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())

    try:
        es.register_grad_reduction_hook_(ddp, mesh_2d, ddp_group="dp")
    except ValueError as error:
        assert "foo" in str(error)
    else:
        raise AssertionError("Expected unknown parameter reduction group to fail at registration")


def test_ddp_grad_reduction_hook_rejects_reduce_group_overlapping_ddp(dist_env, mesh_2d):
    model = nn.Linear(1, 1, bias=False)
    es.set_param_spec(model.weight, es.ParamSpec("o c", reduce="dp"))
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())

    try:
        es.register_grad_reduction_hook_(ddp, mesh_2d, ddp_group="dp")
    except ValueError as error:
        assert "ddp_group" in str(error)
    else:
        raise AssertionError("Expected reduction group overlapping DDP group to fail")


def test_ddp_grad_reduction_hook_validates_combined_groups(dist_env, mesh_2d):
    model = nn.Linear(1, 1, bias=False)
    es.set_param_spec(model.weight, es.ParamSpec("o c", reduce="sp"))
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())

    try:
        es.register_grad_reduction_hook_(
            ddp,
            mesh_2d,
            ddp_group="dp",
            combined_reduce_group="sp-dp",
            combined_reduce="sp",
        )
    except ValueError as error:
        assert "sp-dp" in str(error)
    else:
        raise AssertionError("Expected unknown combined reduction group to fail at registration")


def test_ddp_grad_reduction_hook_rejects_mismatched_combined_group(dist_env, mesh_2d):
    model = nn.Linear(1, 1, bias=False)
    es.set_param_spec(model.weight, es.ParamSpec("o c", reduce="sp"))
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())

    try:
        es.register_grad_reduction_hook_(
            ddp,
            mesh_2d,
            ddp_group="dp",
            combined_reduce_group="sp",
            combined_reduce="sp",
        )
    except ValueError as error:
        assert "combined_reduce_group" in str(error)
    else:
        raise AssertionError("Expected mismatched combined reduction group to fail")


def test_ddp_grad_reduction_hook_rejects_combined_reduce_overlap(dist_env, mesh_2d):
    model = nn.Linear(1, 1, bias=False)
    es.set_param_spec(model.weight, es.ParamSpec("o c", reduce="sp"))
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())

    try:
        es.register_grad_reduction_hook_(
            ddp,
            mesh_2d,
            ddp_group="dp",
            combined_reduce_group="dp",
            combined_reduce="dp",
        )
    except ValueError as error:
        assert "combined_reduce" in str(error)
        assert "overlap" in str(error)
    else:
        raise AssertionError("Expected combined reduction overlapping DDP group to fail")
