import torch
import pytest
import torch_einshard as es


def test_contract_permute():
    torch.manual_seed(333)

    x = torch.randn(4,5,8,6)
    y = torch.randn(5,6,8,3)
    z = es.einshard('a b k c, b c l d -> k l d a', x, y)
    zz = torch.einsum('abkc,bcld->klda', x, y)

    torch.testing.assert_close(z, zz)


def test_outer():
    torch.manual_seed(333)

    x = torch.randn(3)
    y = torch.randn(4)
    z = es.einshard('i, j -> i j', x, y)
    zz = torch.einsum('i,j->ij', x, y)

    torch.testing.assert_close(z, zz)


def test_diagonal():
    torch.manual_seed(333)

    x = torch.randn(5,5)
    z = es.einshard('i i -> i', x)
    zz = torch.einsum('ii->i', x)

    torch.testing.assert_close(z, zz)


def test_ellipsis_contract():
    x = torch.randn(2, 3, 4)
    w = torch.randn(4, 5)

    z = es.einshard("... c, c o -> ... o", x, w)
    torch.testing.assert_close(z, torch.einsum("...c,co->...o", x, w))


def test_ellipsis_reduction():
    x = torch.randn(2, 3, 4)

    z = es.einshard("... c -> c", x)
    torch.testing.assert_close(z, torch.einsum("...c->c", x))


def test_ellipsis_permutation():
    x = torch.randn(2, 3, 4, 5)

    z = es.einshard("b ... c -> ... b c", x)
    torch.testing.assert_close(z, torch.einsum("b...c->...bc", x))


def test_ellipsis_with_factored_axis():
    x = torch.randn(2, 3, 12, 5)
    z = es.einshard("... (h p) c -> ... h p c", x, sizes={"p": 4})
    torch.testing.assert_close(z, x.reshape(2, 3, 3, 4, 5))


def test_axis_families_window_partition_2d():
    x = torch.randn(2, 3, 8, 10, 5)
    families = {"spatial": ("h", "w"), "window": ("wh", "ww")}
    sizes = {"window": (4, 5)}

    z = es.einshard(
        "b t [*spatial *window] c -> (b *spatial) t *window c",
        x,
        families=families,
        sizes=sizes,
    )

    expected = x.reshape(2, 3, 2, 4, 2, 5, 5).permute(0, 2, 4, 1, 3, 5, 6).reshape(8, 3, 4, 5, 5)
    torch.testing.assert_close(z, expected)


def test_axis_families_window_partition_3d():
    x = torch.randn(2, 3, 8, 10, 6, 5)
    families = {"spatial": ("h", "w", "d"), "window": ("wh", "ww", "wd")}
    sizes = {"window": (4, 5, 3)}

    z = es.einshard(
        "b t [*spatial *window] c -> (b *spatial) t *window c",
        x,
        families=families,
        sizes=sizes,
    )

    expected = x.reshape(2, 3, 2, 4, 2, 5, 2, 3, 5)
    expected = expected.permute(0, 2, 4, 6, 1, 3, 5, 7, 8).reshape(16, 3, 4, 5, 3, 5)
    torch.testing.assert_close(z, expected)


def test_axis_families_window_reverse_3d():
    families = {"spatial": ("h", "w", "d"), "window": ("wh", "ww", "wd")}
    sizes = {"b": 2, "h": 2, "w": 2, "d": 2}
    windows = torch.randn(16, 3, 4, 5, 3, 5)

    z = es.einshard(
        "(b *spatial) t *window c -> b t [*spatial *window] c",
        windows,
        families=families,
        sizes=sizes,
    )

    expected = windows.reshape(2, 2, 2, 2, 3, 4, 5, 3, 5)
    expected = expected.permute(0, 4, 1, 5, 2, 6, 3, 7, 8).reshape(2, 3, 8, 10, 6, 5)
    torch.testing.assert_close(z, expected)


def test_axis_family_sizes_expand_by_family_name():
    x = torch.randn(2, 12, 5)

    z = es.einshard(
        "b (*factors) c -> b *factors c",
        x,
        families={"factors": ("h", "p")},
        sizes={"factors": (3, 4)},
    )
    torch.testing.assert_close(z, x.reshape(2, 3, 4, 5))


def test_axis_family_rejects_mismatched_zipped_lengths():
    x = torch.randn(2, 3, 8, 10, 5)

    with pytest.raises(ValueError, match="matching lengths"):
        es.einshard(
            "b t [*spatial *window] c -> b t *spatial *window c",
            x,
            families={"spatial": ("h", "w"), "window": ("wh",)},
        )


def test_expand_factored_axis():
    x = torch.randn(2, 12, 5)

    z = es.einshard("b (h p) c -> b h p c", x, sizes={"p": 4})

    torch.testing.assert_close(z, x.reshape(2, 3, 4, 5))


def test_pack_factored_axis():
    x = torch.randn(2, 3, 4, 5)

    z = es.einshard("b h p c -> b (h p) c", x)

    torch.testing.assert_close(z, x.reshape(2, 12, 5))


def test_factored_axis_with_permutation():
    x = torch.randn(2, 12, 5)
    z = es.einshard("b (h p) c -> b c h p", x, sizes={"h": 3})

    torch.testing.assert_close(z, x.reshape(2, 3, 4, 5).permute(0, 3, 1, 2))


def test_factored_axis_requires_unique_inference():
    x = torch.randn(2, 12, 5)

    try:
        es.einshard("b (h p) c -> b h p c", x)
    except ValueError as error:
        assert "Cannot infer multiple factor sizes" in str(error)
    else:
        raise AssertionError("Expected ambiguous factor sizes to fail")


def test_factored_axis_allows_sharded_annotation_locally():
    x = torch.randn(2, 12, 5)

    z = es.einshard("b (h/sp p) c -> b h/sp p c", x, sizes={"p": 4})
    torch.testing.assert_close(z, x.reshape(2, 3, 4, 5))


def test_factored_axis_rejects_nonlocal_distributed_transform():
    x = torch.randn(2, 12, 5)

    with pytest.raises(NotImplementedError, match="local reshape operations"):
        es.einshard("b (h p) c -> b h/sp p c", x, sizes={"p": 4})
