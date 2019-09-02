# 08-31-2019;
"""
Test cases for warm.module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import warm.module as mm
import warm.functional as W


def test_lambda():
    f = lambda x: x*2
    m = mm.Lambda(f)
    x = torch.randn(1, 2)
    assert torch.equal(f(x), m(x)), 'lambda did not work correctly.'
    def f(x, w, b=5):
        return x*w+b
    m = mm.Lambda(f, 2, b=1)
    assert torch.equal(f(x, 2, 1), m(x)), 'function with args and kwargs did not work correctly.'
    x = torch.randn(3, 2, 4)
    m = mm.Lambda(W.permute, 'BDC', 'BCD')
    assert list(m(x).shape) == [3, 4, 2], 'lambda permute did not work correctly.'


def test_sequential():
    s = mm.Sequential(
        nn.Linear(1, 2),
        nn.LSTM(2, 3, batch_first=True), # lstm and gru return multiple outputs
        nn.GRU(3, 4, batch_first=True),
        mm.Lambda(W.permute, 'BDC', 'BCD'),
        nn.Conv1d(4, 5, 1), )
    x = torch.randn(3, 2, 1)
    assert list(s(x).shape) == [3, 5, 2]


def test_shortcut():
    l = nn.Linear(1, 1, bias=False)
    nn.init.constant_(l.weight, 2.0)
    s = mm.Shortcut(l)
    x = torch.ones(1, 1)
    assert torch.allclose(s(x), torch.Tensor([3.0]))
