import pytest
import torch
import einshard as es

def test_distributed():
    # distributed
    x = torch.randn(8,5)
    y = torch.randn(5,10)
    z = es.einshard('a b / dp, b / dp c -> a c', x, y)
    print(x.shape,y.shape,z.shape)
