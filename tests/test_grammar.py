import pytest

import torch_einshard as es
from torch_einshard.families import cached_expand_axis_families
from torch_einshard.grammar import parse_sharding, sharding


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


def test_parse_sharding_wraps_parse_errors():
    with pytest.raises(ValueError, match="Invalid einshard expression"):
        parse_sharding("a b ->")


def test_parse_sharding_reuses_cached_result():
    assert parse_sharding("a b -> b a") is parse_sharding("a b -> b a")


def test_cached_axis_family_expansion_accepts_list_values():
    expression = "b [*spatial *window] c -> b *spatial *window c"
    families = {"spatial": ["h", "w"], "window": ["wh", "ww"]}
    sizes = {"window": [4, 5]}

    expanded, expanded_sizes = cached_expand_axis_families(expression, sizes, families)

    assert expanded == "b (h wh) (w ww) c -> b h w wh ww c"
    assert expanded_sizes == {"wh": 4, "ww": 5}


def test_einshard_wraps_parse_errors():
    with pytest.raises(ValueError, match="Invalid einshard expression"):
        es.einshard("a b ->")
