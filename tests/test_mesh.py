import torch
import torch.distributed as dist

import torch_einshard as es

from conftest import assert_close


def test_wrap_mesh_keeps_existing_mesh_dimensions(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)

    assert mesh.mesh_dim_names == mesh_2d.mesh_dim_names
    assert dist.get_world_size(mesh["dp"].get_group()) == mesh_2d.mesh.shape[0]


def test_wrap_mesh_treats_string_mesh_dim_name_as_one_name():
    class Mesh:
        mesh = torch.tensor([0])
        mesh_dim_names = "tp"

    mesh = es.wrap_mesh(Mesh())

    assert mesh.mesh_dim_names == ("tp",)


def test_wrap_mesh_rejects_non_string_mesh_dim_names():
    for names in (0, (1,), torch.tensor([])):
        class Mesh:
            mesh = torch.tensor([0])
            mesh_dim_names = names

        try:
            es.wrap_mesh(Mesh())
        except TypeError as error:
            assert "mesh_dim_names" in str(error)
        else:
            raise AssertionError("Expected non-string mesh dim names to fail")


def test_wrap_mesh_rejects_mesh_dim_name_count_mismatch():
    class Mesh:
        mesh = torch.zeros(1, 1)
        mesh_dim_names = ("tp",)

    try:
        es.wrap_mesh(Mesh())
    except ValueError as error:
        assert "mesh_dim_names" in str(error)
    else:
        raise AssertionError("Expected mesh dim name count mismatch to fail")


def test_wrap_mesh_rejects_duplicate_compound_lookup():
    class Mesh:
        mesh = torch.tensor([0])
        mesh_dim_names = ("dp",)

    mesh = es.wrap_mesh(Mesh())

    try:
        mesh["dp-dp"]
    except ValueError as error:
        assert "repeat" in str(error)
    else:
        raise AssertionError("Expected duplicate compound mesh lookup to fail")


def test_wrap_mesh_rejects_empty_compound_lookup():
    class Mesh:
        mesh = torch.tensor([0])
        mesh_dim_names = ("dp",)

    mesh = es.wrap_mesh(Mesh())

    for name in ("dp-", "-dp", "dp--sp"):
        try:
            mesh[name]
        except ValueError as error:
            assert "empty" in str(error)
        else:
            raise AssertionError("Expected empty compound mesh lookup to fail")


def test_wrap_mesh_rejects_empty_or_duplicate_mesh_dim_names():
    for names in ("", ("tp", "tp"), ("dp-dp",), ("dp-",), ("dp--sp",), ("dp", "dp-sp")):
        class Mesh:
            mesh = torch.tensor([0])
            mesh_dim_names = names

        try:
            es.wrap_mesh(Mesh())
        except ValueError as error:
            assert "mesh_dim_names" in str(error)
        else:
            raise AssertionError("Expected invalid mesh dim names to fail")


def test_wrap_mesh_compound_group(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    group = mesh["dp-sp"].get_group()
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()

    x = torch.tensor(float(world_rank + 1))
    x = es.helpers.all_reduce(x, group)

    assert_close(x, torch.tensor(float(world_size * (world_size + 1) // 2)))


def test_wrap_mesh_caches_compound_groups(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)

    assert mesh["dp-sp"].get_group() is mesh["dp-sp"].get_group()
    assert mesh["dp-sp"].get_group() is mesh["sp-dp"].get_group()
