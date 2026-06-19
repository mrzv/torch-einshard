import pytest
import torch
import torch.distributed as dist

import torch_einshard as es

from conftest import assert_close


def test_local_fft_renames_axis():
    x = torch.randn(2, 5, 3, dtype=torch.complex64)

    z = es.einfft("b x c -> b k c", x, axes={"x": "k"})

    assert_close(z, torch.fft.fftn(x, dim=(1,)))


def test_local_fftn_permute_output():
    x = torch.randn(2, 5, 7, dtype=torch.complex64)

    z = es.einfft("b x y -> ky b kx", x, axes={"x": "kx", "y": "ky"})

    expected = torch.fft.fftn(x, dim=(1, 2)).permute(2, 0, 1)
    assert_close(z, expected)


def test_local_fft_expands_iterable_axis_family():
    x = torch.randn(2, 5, 7, dtype=torch.complex64)

    z = es.einfft(
        "b *spatial -> b *spatial",
        x,
        axes=["spatial"],
        families={"spatial": ("x", "y")},
    )

    assert_close(z, torch.fft.fftn(x, dim=(1, 2)))


def test_inverse_fft():
    x = torch.randn(2, 5, dtype=torch.complex64)

    z = es.einfft("b k -> b x", x, axes={"k": "x"}, inverse=True, norm="ortho")

    assert_close(z, torch.fft.ifftn(x, dim=(1,), norm="ortho"))


def test_rejects_partial_specs():
    x = torch.randn(2, 5, dtype=torch.complex64)

    with pytest.raises(ValueError, match="partial"):
        es.einfft("b x // tp -> b k", x, axes={"x": "k"})


def test_rejects_string_axes():
    x = torch.randn(2, 5, dtype=torch.complex64)

    with pytest.raises(TypeError, match="not a string"):
        es.einfft("b time -> b freq", x, axes="time")


def test_sharded_transform_axis_gather_fft_split(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    size = world_size * 3 + 1
    shapes = es.helpers.compute_split_shapes(size, world_size)

    full = torch.randn(2, size, dtype=torch.complex64)
    x = torch.split(full, shapes, dim=1)[rank].contiguous()

    z = es.einfft(
        "b x/tp -> b k/tp",
        x,
        axes={"x": "k"},
        mesh=mesh_tp,
        shapes={"tp": {"x": shapes, "k": shapes}},
    )

    expected = torch.split(torch.fft.fftn(full, dim=(1,)), shapes, dim=1)[rank]
    assert_close(z, expected)


def test_sharded_transform_axis_permute_output(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    size = world_size * 3 + 1
    shapes = es.helpers.compute_split_shapes(size, world_size)

    full = torch.randn(2, size, dtype=torch.complex64)
    x = torch.split(full, shapes, dim=1)[rank].contiguous()

    z = es.einfft(
        "b x/tp -> k/tp b",
        x,
        axes={"x": "k"},
        mesh=mesh_tp,
        shapes={"tp": {"x": shapes, "k": shapes}},
    )

    expected = torch.split(torch.fft.fftn(full, dim=(1,)).mT, shapes, dim=0)[rank]
    assert_close(z, expected)


def test_sharded_output_axis(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    size = world_size * 3 + 1
    shapes = es.helpers.compute_split_shapes(size, world_size)

    x = torch.randn(2, size, dtype=torch.complex64)

    z = es.einfft(
        "b x -> b k/tp",
        x,
        axes={"x": "k"},
        mesh=mesh_tp,
        shapes=shapes,
    )

    expected = torch.split(torch.fft.fftn(x, dim=(1,)), shapes, dim=1)[rank]
    assert_close(z, expected)
