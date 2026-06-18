import torch
import torch.distributed as dist
import torch.nn.functional as F
import pytest

import torch_einshard as es

from conftest import assert_close


def test_local_einconv_matches_conv2d_with_default_full_checkpoint():
    torch.manual_seed(333)
    x = torch.randn(2, 5, 6, 3, requires_grad=True)
    weight = torch.randn(4, 3, 3, 3, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)

    z = es.einconv(
        "b h w c, o c kh kw -> b h w o",
        x,
        weight,
        {"h": "kh", "w": "kw"},
        bias=bias,
    )
    expected = F.conv2d(x.permute(0, 3, 1, 2), weight, bias, padding=1).permute(0, 2, 3, 1)

    assert_close(z, expected)

    z.square().sum().backward()
    x_grad = x.grad.clone()
    weight_grad = weight.grad.clone()
    bias_grad = bias.grad.clone()

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    expected_ref = F.conv2d(x_ref.permute(0, 3, 1, 2), weight_ref, bias_ref, padding=1).permute(0, 2, 3, 1)
    expected_ref.square().sum().backward()

    assert_close(x_grad, x_ref.grad, rtol=1e-4, atol=1e-5)
    assert_close(weight_grad, weight_ref.grad, rtol=1e-4, atol=1e-5)
    assert_close(bias_grad, bias_ref.grad, rtol=1e-4, atol=1e-5)


def test_local_einconv_supports_conv_checkpoint_mode():
    x = torch.randn(2, 5, 6, 3, requires_grad=True)
    weight = torch.randn(4, 3, 3, 3, requires_grad=True)

    z = es.einconv(
        "b h w c, o c kh kw -> b h w o",
        x,
        weight,
        {"h": "kh", "w": "kw"},
        checkpoint="conv",
    )

    expected = F.conv2d(x.permute(0, 3, 1, 2), weight, padding=1).permute(0, 2, 3, 1)
    assert_close(z, expected)


def test_local_einconv_axis_families():
    x = torch.randn(2, 5, 6, 3)
    weight = torch.randn(4, 3, 3, 3)

    z = es.einconv(
        "b *spatial c, o c *window -> b *spatial o",
        x,
        weight,
        {"spatial": ("kh", "kw")},
        families={"spatial": ("h", "w"), "window": ("kh", "kw")},
        checkpoint=False,
    )

    expected = F.conv2d(x.permute(0, 3, 1, 2), weight, padding=1).permute(0, 2, 3, 1)
    assert_close(z, expected)


def test_einconv_rejects_length_changing_padding():
    x = torch.randn(2, 5, 6, 3)
    weight = torch.randn(4, 3, 3, 3)

    with pytest.raises(ValueError, match="preserve the spatial length"):
        es.einconv(
            "b h w c, o c kh kw -> b h w o",
            x,
            weight,
            {"h": "kh", "w": "kw"},
            padding={"h": 0, "w": 1},
        )


def test_einconv_rejects_groups_until_notation_is_defined():
    x = torch.randn(2, 5, 6, 4)
    weight = torch.randn(4, 2, 3, 3)

    with pytest.raises(NotImplementedError, match="groups=1"):
        es.einconv(
            "b h w c, o c kh kw -> b h w o",
            x,
            weight,
            {"h": "kh", "w": "kw"},
            groups=2,
        )


def test_distributed_einconv_matches_full_conv2d(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    shapes = es.helpers.compute_split_shapes(world_size * 3 + 1, world_size)
    full = torch.randn(2, sum(shapes), 5, 3)
    weight = torch.randn(4, 3, 3, 3)
    bias = torch.randn(4)
    x = torch.split(full, shapes, dim=1)[rank].contiguous().requires_grad_(True)

    z = es.einconv(
        "b h/dp w c, o c kh kw -> b h/dp w o",
        x,
        weight,
        {"h": "kh", "w": "kw"},
        bias=bias,
        mesh=mesh_1d,
        shapes=shapes,
    )

    expected = F.conv2d(full.permute(0, 3, 1, 2), weight, bias, padding=1).permute(0, 2, 3, 1)
    expected = torch.split(expected, shapes, dim=1)[rank]
    assert_close(z, expected)

    z.square().sum().backward()

    full_ref = full.detach().clone().requires_grad_(True)
    expected_ref = F.conv2d(full_ref.permute(0, 3, 1, 2), weight, bias, padding=1).permute(0, 2, 3, 1)
    expected_ref.square().sum().backward()
    expected_grad = torch.split(full_ref.grad, shapes, dim=1)[rank]
    assert_close(x.grad, expected_grad, rtol=1e-4, atol=1e-5)
