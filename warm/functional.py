# -*- coding: utf-8 -*-
# 08-27-2019;
"""
"""
import torch
import torch.nn as nn
import numpy as np
from . import engine


permute = engine.permute


def conv(x, size, kernel, init_weight=None, init_bias=None, bias=True, **kw):
    """ """
    d = x.ndim-3
    assert d in [0, 1, 2], 'Incompatible number of dims for input x.'
    inferred_kw = dict(
        base_name='conv',
        base_class=[nn.Conv1d, nn.Conv2d, nn.Conv3d][d],
        base_kw={
            'out_channels':size,
            'kernel_size':kernel,
            'bias':bias,
            **engine.unused_kwargs(kw), },
        infer_kw={'in_channels':'C'},
        initialization={'weight':init_weight, **({'bias':init_bias} if bias else {})}, )
    return engine.forward(x, **{**inferred_kw, **kw})


def linear(x, size, init_weight=None, init_bias=None, bias=True, **kw):
    """ """
    inferred_kw = dict(
        base_name='linear',
        base_class=nn.Linear,
        base_kw={'out_features':size, 'bias':bias},
        base_shape='BDC',
        infer_kw={'in_features':'C'},
        initialization={'weight':init_weight, **({'bias':init_bias} if bias else {})}, )
    return engine.forward(x, **{**inferred_kw, **kw})


def batch_norm(x, **kw):
    """ """
    d = x.ndim-3
    assert d in [0, 1, 2], 'Incompatible number of dims for input x.'
    inferred_kw = dict(
        base_name='batch_norm',
        base_class=[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][d],
        base_kw={'num_features':x.shape[1]}, )
    return engine.forward(x, **{**inferred_kw, **kw})


def lstm(x, size,
        init_weight_hh='orthogonal_', init_weight_ih=None, init_bias_hh=None, init_bias_ih=None,
        bias=True, num_layers=1, **kw):
    """ """
    init = dict(
        weight_hh=init_weight_hh,
        weight_ih=init_weight_ih,
        bias_hh=init_bias_hh,
        bias_ih=init_bias_ih, )
    inferred_kw = dict(
        base_name='lstm',
        base_class=nn.LSTM,
        base_kw={
            'hidden_size':size,
            'num_layers':num_layers,
            **engine.unused_kwargs(kw), },
        base_shape='DBC',
        infer_kw={'input_size':'C'},
        initialization={
            f'{k}_l{l}':init[k] for k in ['weight_hh', 'weight_ih']+(['bias_hh', 'bias_ih'] if bias else [])
            for l in range(num_layers)}, )
    return engine.forward(x, **{**inferred_kw, **kw})


def gru(*arg, **kw):
    """ """
    return lstm(*arg, base_name='gru', base_class=nn.GRU, **kw)


def identity(x, *arg, **kw):
    return x


# def embedding(x, num_embeddings, embedding_dim, init_weight=None, **kw):
#     inferred_kw = dict(
#         base_name='embedding',
#         base_class=nn.Embedding,
#         base_kw={
#             'num_embeddings':num_embeddings,
#             'embedding_dim':embedding_dim,
#             **engine.unused_kwargs(kw), },
#         base_shape='BDC',
#         initialization={'weight':init_weight}, )
#     return engine.forward(x, **{**inferred_kw, **kw})
