# -*- coding: utf-8 -*-
# blue-season; 08-27-2019;
"""
"""
import torch
import torch.nn as nn
import numpy as np
from . import engine


def conv(x, size, kernel, init_weight=None, init_bias=None, name=None, parent=None, activation=None, **kw):
    d = x.ndim-3
    assert d in [0, 1, 2], 'Incompatible number of dims for input x.'
    forward_kw = dict(
        base_name='conv',
        base_class=[nn.Conv1d, nn.Conv2d, nn.Conv3d][d],
        base_kw={'in_channels':x.shape[1], 'out_channels':size, 'kernel_size':kernel, **kw},
        activation=activation,
        initialization={'weight':init_weight, 'bias':init_bias},
        parent=parent,
        name=name, )
    return engine.forward(x, **{**forward_kw, **kw})


def linear(x, size, init_weight=None, init_bias=None, name=None, parent=None, activation=None, bias=True, **kw):
    forward_kw = dict(
        base_name='linear',
        base_class=nn.Linear,
        base_kw={'in_features':x.shape[-1], 'out_features':size, 'bias':bias},
        base_shape='BDC',
        in_shape='BCD',
        out_shape='BCD',
        activation=activation,
        initialization={'weight':init_weight, 'bias':init_bias},
        parent=parent,
        name=name, )
    return engine.forward(x, **{**forward_kw, **kw})


def batch_norm():
    pass


def lstm():
    pass


