# 08-31-2019;
"""
Test cases for warm.functional.
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import warm.module as mm
import warm.functional as W


def test_conv():
    m = nn.Module()
    x = torch.randn(1, 2, 8) # BCD
    torch.manual_seed(100)
    y0 = nn.Conv1d(2, 3, 3)(x)
    torch.manual_seed(100)
    y1 = W.conv(x, 3, 3, parent=m)
    assert torch.equal(y0, y1), 'conv incorrect output on 1d signal.'
    m = nn.Module()
    x = torch.randn(1, 2, 3, 4) # BCD
    torch.manual_seed(100)
    y0 = nn.Conv2d(2, 3, 3)(x)
    torch.manual_seed(100)
    y1 = W.conv(x, 3, 3, parent=m)
    assert torch.equal(y0, y1), 'conv incorrect output on 2d signal.'


def test_linear():
    m = nn.Module()
    x = torch.randn(1, 2, 3) # BDC
    torch.manual_seed(100)
    y0 = nn.Linear(3, 4)(x)
    torch.manual_seed(100)
    y1 = W.linear(x, 4, parent=m, in_shape='BDC', out_shape='BDC')
    assert torch.equal(y0, y1), 'linear incorrect output on 1d signal.'
    m = nn.Module()
    x = torch.randn(1, 2, 3, 4) # BDC
    torch.manual_seed(100)
    y0 = nn.Linear(4, 3)(x)
    torch.manual_seed(100)
    y1 = W.linear(x, 3, parent=m, in_shape='BDC', out_shape='BDC')
    assert torch.equal(y0, y1), 'batch_norm incorrect output on 2d signal.'


def test_batch_norm():
    m = nn.Module()
    x = torch.randn(1, 2, 3) # BCD
    torch.manual_seed(100)
    y0 = nn.BatchNorm1d(2)(x)
    torch.manual_seed(100)
    y1 = W.batch_norm(x, parent=m)
    m = nn.Module()
    assert torch.equal(y0, y1), 'batch_norm incorrect output on 1d signal.'
    x = torch.randn(1, 2, 3, 4) # BCD
    torch.manual_seed(100)
    y0 = nn.BatchNorm2d(2)(x)
    torch.manual_seed(100)
    y1 = W.batch_norm(x, parent=m)
    assert torch.equal(y0, y1), 'batch_norm incorrect output on 2d signal.'


def test_lstm():
    m = nn.Module()
    x = torch.randn(3, 2, 1) # DBC
    torch.manual_seed(100)
    y0, *_ = nn.LSTM(1, 2, num_layers=2)(x)
    torch.manual_seed(100)
    y1 = W.lstm(x, 2, num_layers=2, parent=m, init_weight_hh=None, in_shape='DBC', out_shape='DBC')
    assert torch.equal(y0, y1)


def test_gru():
    m = nn.Module()
    x = torch.randn(3, 2, 1) # DBC
    torch.manual_seed(100)
    y0, *_ = nn.GRU(1, 2, num_layers=2)(x)
    torch.manual_seed(100)
    y1 = W.gru(x, 2, num_layers=2, parent=m, init_weight_hh=None, in_shape='DBC', out_shape='DBC')
    assert torch.equal(y0, y1)


def test_identity():
    x = torch.randn(1, 2, 3)
    assert torch.equal(W.identity(x, 7, 8, a='b'), x)


def test_dropout():
    m = nn.Module()
    x = torch.ones(2, 6, 6, 6)
    torch.manual_seed(100)
    y0 = nn.Dropout(0.3)(x)
    torch.manual_seed(100)
    y1 = W.dropout(x, 0.3, parent=m)
    assert torch.equal(y0, y1)
    torch.manual_seed(100)
    y0 = nn.Dropout2d(0.3)(x)
    torch.manual_seed(100)
    y1 = W.dropout(x, 0.3, by_channel=True, parent=m)
    assert torch.equal(y0, y1)


def test_transformer():
    m = nn.Module()
    x = torch.randn(10, 2, 4)
    y = torch.randn(6, 2, 4)
    torch.manual_seed(100)
    z0 = nn.Transformer(4, 2, 1, 1, dim_feedforward=8)(x, y)
    torch.manual_seed(100)
    z1 = W.transformer(x, y, 1, 1, 2, dim_feedforward=8, in_shape='DBC', out_shape='DBC', parent=m)
    assert torch.equal(z0, z1)
    torch.manual_seed(100)
    z1 = W.transformer(x, y, 1, 1, 2, dim_feedforward=8, in_shape='DBC', out_shape='DBC', parent=m, causal=True)
    assert not torch.equal(z0, z1)
    z1 = W.transformer(x, None, 2, 0, 2, dim_feedforward=8, in_shape='DBC', out_shape='DBC', parent=m)
    assert z1.shape == x.shape
