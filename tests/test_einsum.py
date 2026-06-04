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
