import torch
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
