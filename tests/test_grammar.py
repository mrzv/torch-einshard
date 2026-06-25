import pytest

import torch_einshard as es
from torch_einshard.families import cached_expand_axis_families
from torch_einshard.grammar import parse_sharding, sharding
from torch_einshard.sharding import Axis


def test_parse_single_partial():
    x, y, z = sharding("b n h // tp -> b n h").map()

    assert y is None
    assert [axis.name for axis in x.axes] == ["b", "n", "h"]
    assert x.partials == ("tp",)
    assert [axis.name for axis in z.axes] == ["b", "n", "h"]
    assert z.partials == ()


def test_parse_multiple_partials():
    x, y, z = sharding("loss // (sp1,sp2) -> loss").map()

    assert y is None
    assert [axis.name for axis in x.axes] == ["loss"]
    assert x.partials == ("sp1", "sp2")
    assert [axis.name for axis in z.axes] == ["loss"]
    assert z.partials == ()


def test_parse_shard_to_partial():
    x, y, z = sharding("b n/tp h -> b n h // tp").map()

    assert y is None
    assert x.axes[1].name == "n"
    assert x.axes[1].shard_dim == "tp"
    assert z.partials == ("tp",)


def test_parse_axis_group():
    x, y, z = sharding("b (h p) c -> b h p c").map()

    assert y is None
    assert [axis.name for axis in x.axes[1].axes] == ["h", "p"]
    assert [axis.name for axis in x.axes.flat()] == ["b", "h", "p", "c"]
    assert [axis.name for axis in z.axes] == ["b", "h", "p", "c"]


def test_parse_ellipsis_axis():
    x, y, z = sharding("... c -> ... c").map()

    assert y is None
    assert [axis.name for axis in x.axes] == ["...", "c"]
    assert [axis.name for axis in z.axes] == ["...", "c"]


def test_parse_rejects_multiple_ellipsis_axes():
    with pytest.raises(ValueError, match="at most one ellipsis"):
        sharding("... a ... -> a").map()


def test_parse_sharded_axis_in_group():
    x, y, z = sharding("b (h/sp p) c -> b h/sp p c").map()

    assert y is None
    assert x.axes[1].axes[0].name == "h"
    assert x.axes[1].axes[0].shard_dim == "sp"
    assert z.axes[1].shard_dim == "sp"


def test_parse_hyphenated_mesh_dimension_name():
    x, y, z = sharding("b n/tp-sp h -> b n h // tp-sp").map()

    assert y is None
    assert x.axes[1].name == "n"
    assert x.axes[1].shard_dim == "tp-sp"
    assert z.partials == ("tp-sp",)


def test_parse_hyphenated_mesh_dimension_in_partial_list():
    x, y, z = sharding("loss // (sp1-sp2,dp-sp1-sp2) -> loss").map()

    assert y is None
    assert x.partials == ("sp1-sp2", "dp-sp1-sp2")
    assert z.partials == ()


def test_parse_parameter_annotation_with_async_grad():
    x, y, z = parse_sharding("b c, out/tp c [param, grad=async] -> b out/tp")

    assert not x.annotation
    assert y.annotation.is_param
    assert y.annotation.grad.mode == "inferred"
    assert y.annotation.grad.mesh_dims == ()
    assert y.annotation.grad.backend == "native"
    assert y.annotation.grad.schedule == "async"
    assert not z.annotation


def test_parse_parameter_annotation_with_explicit_grad_backend():
    _, y, _ = parse_sharding("b c, out c [param, grad=dp:ddp] -> b out")

    assert y.annotation.grad.mode == "explicit"
    assert y.annotation.grad.mesh_dims == ("dp",)
    assert y.annotation.grad.backend == "ddp"
    assert y.annotation.grad.schedule == "backend_default"


def test_parse_parameter_annotation_with_external_grad_and_init_sync():
    _, y, _ = parse_sharding("b c, out c [param, grad=external, init_sync=none] -> b out")

    assert y.annotation.grad.mode == "inferred"
    assert y.annotation.grad.backend == "external"
    assert y.annotation.init_sync.mode == "none"
    assert y.annotation.init_sync.mesh_dims == ()


def test_parse_annotation_repr_round_trips_through_cache_copy():
    parsed = parse_sharding("b c, out/tp c [param, grad=sp1-sp2:async, init_sync=tp] -> b out/tp")
    reparsed = parse_sharding("b c, out/tp c [param, grad=sp1-sp2:async, init_sync=tp] -> b out/tp")

    assert parsed is not reparsed
    assert repr(parsed) == repr(reparsed)
    assert repr(parsed[1]) == "out / tp × c [param, grad=sp1-sp2:async, init_sync=tp]"


def test_parse_rejects_standalone_async_annotation():
    with pytest.raises(ValueError, match="Invalid einshard expression"):
        parse_sharding("a [async] -> a")


def test_parse_rejects_output_annotation():
    with pytest.raises(ValueError, match="Invalid einshard expression"):
        parse_sharding("a -> a [param]")


def test_parse_rejects_duplicate_annotation_items():
    with pytest.raises(ValueError, match="Duplicate tensor annotation"):
        parse_sharding("a [param, param] -> a")


def test_parse_sharding_wraps_parse_errors():
    with pytest.raises(ValueError, match="Invalid einshard expression"):
        parse_sharding("a b ->")


def test_parse_sharding_returns_copy_of_cached_result():
    parsed = parse_sharding("a b -> b a")
    reparsed = parse_sharding("a b -> b a")

    assert parsed is not reparsed
    assert repr(parsed) == repr(reparsed)


def test_parse_sharding_mutation_does_not_poison_cache():
    parsed = parse_sharding("a -> a")
    parsed[0].axes.append(Axis("b"))

    reparsed = parse_sharding("a -> a")

    assert [axis.name for axis in reparsed[0].axes] == ["a"]


def test_cached_axis_family_expansion_accepts_list_values():
    expression = "b [*spatial *window] c -> b *spatial *window c"
    families = {"spatial": ["h", "w"], "window": ["wh", "ww"]}
    sizes = {"window": [4, 5]}

    expanded, expanded_sizes = cached_expand_axis_families(expression, sizes, families)

    assert expanded == "b (h wh) (w ww) c -> b h w wh ww c"
    assert expanded_sizes == {"wh": 4, "ww": 5}


def test_axis_family_expansion_ignores_parameter_annotations():
    expression = "b [*spatial *window] c, out c [param, grad = async] -> b *spatial *window out"
    families = {"spatial": ("h", "w"), "window": ("wh", "ww")}

    expanded, _ = cached_expand_axis_families(expression, None, families)
    _, weight, _ = parse_sharding(expanded)

    assert expanded == "b (h wh) (w ww) c, out c [param, grad = async] -> b h w wh ww out"
    assert weight.annotation.is_param
    assert weight.annotation.grad.schedule == "async"


def test_cached_axis_family_expansion_returns_copy_of_sizes():
    expression = "b *spatial c -> b *spatial c"
    families = {"spatial": ("h", "w")}
    sizes = {"spatial": (4, 5)}

    _, expanded_sizes = cached_expand_axis_families(expression, sizes, families)
    expanded_sizes["h"] = 99
    _, reparsed_sizes = cached_expand_axis_families(expression, sizes, families)

    assert reparsed_sizes == {"h": 4, "w": 5}


def test_einshard_wraps_parse_errors():
    with pytest.raises(ValueError, match="Invalid einshard expression"):
        es.einshard("a b ->")
