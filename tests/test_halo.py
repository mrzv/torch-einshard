import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import torch_einshard as es

from conftest import assert_close


def _extended_split(full, shapes, rank, left, right, *, fill=0):
    start = sum(shapes[:rank])
    end = start + shapes[rank]
    take_start = max(0, start - left)
    take_end = min(full.shape[0], end + right)
    result = full[take_start:take_end]
    if take_start > start - left or take_end < end + right:
        result = F.pad(result, (0, 0, take_start - (start - left), (end + right) - take_end), value=fill)
    return result


def test_local_einhalo_constant_boundary():
    x = torch.arange(5, dtype=torch.float32)

    z = es.einhalo("h", x, {"h": (2, 1)}, fill=-1)

    assert_close(z, torch.tensor([-1, -1, 0, 1, 2, 3, 4, -1], dtype=torch.float32))


def test_local_einhalo_periodic_boundary():
    x = torch.arange(5, dtype=torch.float32)

    z = es.einhalo("h", x, {"h": (2, 1)}, boundary="periodic")

    assert_close(z, torch.tensor([3, 4, 0, 1, 2, 3, 4, 0], dtype=torch.float32))


def test_local_einhalo_replicate_boundary():
    x = torch.arange(5, dtype=torch.float32)

    z = es.einhalo("h", x, {"h": (2, 1)}, boundary="replicate")

    assert_close(z, torch.tensor([0, 0, 0, 1, 2, 3, 4, 4], dtype=torch.float32))


def test_einhalo_rejects_unknown_axis():
    x = torch.randn(3)

    with pytest.raises(ValueError, match="Halo axis 'w'"):
        es.einhalo("h", x, {"w": 1})


def test_einhalo_rejects_empty_periodic_axis():
    x = torch.empty(0)

    with pytest.raises(ValueError, match="periodic-pad an empty axis"):
        es.einhalo("h", x, {"h": 1}, boundary="periodic")


def test_einhalo_rejects_factored_axes():
    x = torch.randn(6)

    with pytest.raises(NotImplementedError, match="factored axes"):
        es.einhalo("(h p)", x, {"h": 1})


def test_einhalo_rejects_duplicate_axis_names():
    x = torch.randn(3, 3)

    with pytest.raises(ValueError, match="appears more than once"):
        es.einhalo("h h", x, {"h": 1})


def test_local_einwindow_adds_explicit_window_axis():
    x = torch.arange(5, dtype=torch.float32)

    z = es.einwindow("h -> h kh", x, {"h": "kh"}, {"h": 1}, fill=-1)

    expected = torch.tensor(
        [
            [-1, 0, 1],
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, -1],
        ],
        dtype=torch.float32,
    )
    assert_close(z, expected)


def test_einwindow_rejects_output_sharding_changes():
    x = torch.randn(5)

    with pytest.raises(ValueError, match="preserve input axis sharding"):
        es.einwindow("h -> h/dp kh", x, {"h": "kh"}, {"h": 1})

    with pytest.raises(ValueError, match="window axes must be local"):
        es.einwindow("h -> h kh/dp", x, {"h": "kh"}, {"h": 1})


def test_einwindow_rejects_duplicate_axis_names():
    x = torch.randn(3, 3)

    with pytest.raises(ValueError, match="appears more than once"):
        es.einwindow("h h -> h h kh", x, {"h": "kh"}, {"h": 1})


def test_einwindow_rejects_extra_radius_axis():
    x = torch.randn(5)

    with pytest.raises(ValueError, match="Radius axis 'w'"):
        es.einwindow("h -> h kh", x, {"h": "kh"}, {"h": 1, "w": 1})


def test_einwindow_supports_convolution_style_contraction():
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4, 1)
    weight = torch.ones(1, 3, 3, 1)

    patches = es.einwindow(
        "b h w c -> b h w kh kw c",
        x,
        {"h": "kh", "w": "kw"},
        {"h": 1, "w": 1},
    )
    z = es.einshard("b h w kh kw c, o kh kw c -> b h w o", patches, weight)

    expected = F.conv2d(x.permute(0, 3, 1, 2), weight.permute(0, 3, 1, 2), padding=1)
    assert_close(z, expected.permute(0, 2, 3, 1))


def test_distributed_einhalo_uneven_shards(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    shapes = es.helpers.compute_split_shapes(world_size * 3 + 1, world_size)
    full = torch.arange(sum(shapes) * 2, dtype=torch.float32).reshape(sum(shapes), 2)
    x = torch.split(full, shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einhalo("a/dp b", x, {"a": 1}, mesh=mesh_1d, shapes=shapes, fill=-1)

    assert_close(z, _extended_split(full, shapes, rank, 1, 1, fill=-1))

    z.sum().backward()
    full_grad = torch.zeros_like(full)
    for peer in range(world_size):
        start = sum(shapes[:peer])
        end = start + shapes[peer]
        full_grad[max(0, start - 1):min(full.shape[0], end + 1)] += 1
    assert_close(x.grad, torch.split(full_grad, shapes, dim=0)[rank])


def test_distributed_einhalo_rejects_empty_periodic_axis(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    world_size = dist.get_world_size(group)
    x = torch.empty(0, 2)

    with pytest.raises(ValueError, match="periodic-pad an empty axis"):
        es.einhalo("a/dp b", x, {"a": 1}, boundary="periodic", mesh=mesh_1d, shapes=[0] * world_size)


def test_distributed_einwindow_matches_full_tensor_windows(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    shapes = es.helpers.compute_split_shapes(world_size * 3 + 1, world_size)
    full = torch.arange(sum(shapes), dtype=torch.float32).reshape(sum(shapes), 1)
    x = torch.split(full, shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einwindow(
        "a/dp b -> a/dp k b",
        x,
        {"a": "k"},
        {"a": 1},
        mesh=mesh_1d,
        shapes=shapes,
        fill=-1,
    )

    expected = _extended_split(full, shapes, rank, 1, 1, fill=-1).unfold(0, 3, 1).permute(0, 2, 1)
    assert_close(z, expected)


def test_einwindow_axis_family():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    z = es.einwindow(
        "*spatial -> *spatial *window",
        x,
        {"spatial": ("kh", "kw")},
        {"spatial": (1, 0)},
        families={"spatial": ("h", "w"), "window": ("kh", "kw")},
    )

    assert z.shape == (3, 4, 3, 1)
