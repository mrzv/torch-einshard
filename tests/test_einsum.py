import pytest
import torch
import einshard as es

def test_contract_permute():
    # contract + permute
    x = torch.randn(4,5,8,6)
    y = torch.randn(5,6,8,3)
    z = es.einshard('a b k c, b c l d -> k l d a', x, y)
    zz = torch.einsum('abkc,bcld->klda', x, y)
    print(x.shape,y.shape,z.shape,zz.shape)
    assert torch.norm(z - zz) == 0.

def test_outer():
    # outer product
    x = torch.randn(3)
    y = torch.randn(4)
    z = es.einshard('i, j -> i j', x, y)
    zz = torch.einsum('i,j->ij', x, y)
    print(x.shape,y.shape,z.shape,zz.shape)
    assert torch.norm(z - zz) == 0.

def test_diagonal():
    # diagonal
    x = torch.randn(5,5)
    z = es.einshard('i i -> i', x)
    zz = torch.einsum('ii->i', x)
    print(x.shape,z.shape,zz.shape)
    assert torch.norm(z - zz) == 0.
