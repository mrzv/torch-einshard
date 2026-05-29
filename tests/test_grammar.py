from torch_einshard.grammar import sharding


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
