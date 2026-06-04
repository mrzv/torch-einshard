import torch
import torch.distributed as dist

import torch_einshard as es

from conftest import assert_close


def test_wrap_mesh_keeps_existing_mesh_dimensions(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)

    assert mesh.mesh_dim_names == mesh_2d.mesh_dim_names
    assert dist.get_world_size(mesh["dp"].get_group()) == mesh_2d.mesh.shape[0]


def test_wrap_mesh_compound_group(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    group = mesh["dp-sp"].get_group()
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()

    x = torch.tensor(float(world_rank + 1))
    es.helpers.all_reduce(x, group)

    assert_close(x, torch.tensor(float(world_size * (world_size + 1) // 2)))


def test_wrap_mesh_caches_compound_groups(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)

    assert mesh["dp-sp"].get_group() is mesh["dp-sp"].get_group()
    assert mesh["dp-sp"].get_group() is mesh["sp-dp"].get_group()
