# 08-27-2019;
"""
Wraps around various torch.nn Modules to fit into a functional interface.
"""
import torch.nn as nn
from warm import engine


permute = engine.permute


def conv(x, size, kernel, init_weight=None, init_bias=None, bias=True, **kw):
    """ Convolution layer.\n
    - `x: Tensor`; With shape `(Batch, Channel, *)` where `*` Can be 1d or 2d or 3d.
        If 3d, shapes are `(Batch, Channel, Length)`.
        If 4d, shapes are `(Batch, Channel, Height, Width)`.
        If 5d, shapes are `(Batch, Channel, Depth, Height, Width)`.
    - `size: int`; Size of hidden filters, and size of the output channel.
    - `kernel: int or tuple`; Size of the convolution kernel.
    - `init_weight: None or str or callable`; Initialization specification for the weight tensor.
        If a `str`, should be one of the nonlinearity functions contained in `torch.nn.init`.
        If a `callable`, it will be applied to `x` directly, i.e. `spec(x)`. If a 2-`tuple`,
        it must be of format `(callable, kwargs)`, i.e. `callable(x, **kwargs)`.
        Default: `None`, and the weight tensor is initialized using `torch.nn.ConvNd`s default scheme.
    - `init_bias: None or str or callable`; Same as `init_weight`, but for the bias tensor.
    - `bias: bool`; If `True`, adds a learnable bias to the output. Default: `True`.
    - `**kw:dict`; Any additional KWargs are passed down to `torch.nn.ConvNd`, where N can be 1, 2 or 3.
        as well as `warm.engine.forward`. Refer to their docs for details. Some of the additional ConvNd arguments:
        `stride, padding, dilation, groups`.
    - `return: Tensor`; With shape `(Batch, Size, *)` where `*` can be 1d, 2d, 3d that depends on `x`. """
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
    """ Linear transformation layer.\n
    - `x: Tensor`; 2d or more, with shapes `(Batch, Channel, *)` where `*` means any number of additional dimensions.
    - `size: int`; Size of hidden features, and size of the output channel.
    - `init_weight: None or str or callable`; Initialization specification for the weight tensor.
        If a `str`, should be one of the nonlinearity functions contained in `torch.nn.init`.
        If a `callable`, it will be applied to `x` directly, i.e. `spec(x)`. If a 2-`tuple`,
        it must be of format `(callable, kwargs)`, i.e. `callable(x, **kwargs)`.
        Default: `None`, and the weight tensor is initialized using `torch.nn.Linear`s default scheme.
    - `init_bias: None or str or callable`; Same as `init_weight`, but for the bias tensor.
    - `bias: bool`; If `True`, adds a learnable bias to the output. Default: `True`.
    - `**kw:dict`; Any additional KWargs are passed down to `warm.engine.forward`. Refer to its docs for details.
    - `return: Tensor`; With shape `(Batch, Size, *)` where `*` can be 1d, 2d, 3d that depends on `x`. """
    inferred_kw = dict(
        base_name='linear',
        base_class=nn.Linear,
        base_kw={'out_features':size, 'bias':bias},
        base_shape='BDC',
        infer_kw={'in_features':'C'},
        initialization={'weight':init_weight, **({'bias':init_bias} if bias else {})}, )
    return engine.forward(x, **{**inferred_kw, **kw})


def batch_norm(x, **kw):
    """ Batch Normalization layer.\n
    - `x: Tensor`; 2d or more, with shapes `(Batch, Channel, *)` where `*` means any number of additional dimensions.
    - `**kw: dict`; Any additional KWargs are passed down to `torch.nn.BatchNormNd`, where N can be 1, 2 or 3.
        as well as `warm.engine.forward`. Refer to their docs for details. Some of the additional BatchNorm arguments:
        `eps, momentum, affine, track_running_stats`.
    - `return: Tensor`; Same shape as input  `x`. """
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
    """ Long Short Term Memory layer.\n
    - `x: Tensor`; 3d, with shapes `(Batch, Channel, Length)`.
    - `size: int`; Size of hidden features, and size of the output channel.
    - `init_weight_hh: None or str or callable`; Initialization specification for the hidden-hidden weight tensor.
        If a `str`, should be one of the nonlinearity functions contained in `torch.nn.init`.
        If a `callable`, it will be applied to `x` directly, i.e. `spec(x)`. If a 2-`tuple`,
        it must be of format `(callable, kwargs)`, i.e. `callable(x, **kwargs)`.
        Default: `'orthogonal_'`.
    - `init_weight_ih: None or str or callable`; Initialization specification for the input-hidden weight tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.LSTM`s default scheme.
    - `init_bias_hh: None or str or callable`; Initialization specification for the hidden-hidden bias tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.LSTM`s default scheme.
    - `init_bias_ih: None or str or callable`; Initialization specification for the input-hidden bias tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.LSTM`s default scheme.
    - `bias: bool`; If `False`, then the layer does not use `bias_ih` and `bias_hh`. Default: `True`.
    - `num_layers: int`; Number of the recurrent layers. Default: 1.
    - `tuple_out: bool`; If `True`, the returned value will be a tuple `(out, (h_n, c_n))`. Default: False.
    - `**kw: dict`; Any additional KWargs are passed down to `torch.nn.LSTM`, as well as `warm.engine.forward`.
        Refer to their docs for details. Some of the additional LSTM arguments: `dropout, bidirectional, batch_first`.
    - `return: Tensor or tuple`; If `tuple_out` set to true, will return `(out, (h_n, c_n)`, otherwise just `out`.
        `out` has shape `(Batch, Size, Length*Directions)`, where Directions = 2 if `bidirectional` else 1.
        `h_n` has shape `(num_layers*Directions, Batch, Size)`; The hidden states. 
        `c_n` has shape `(num_layers*Directions, Batch, Size)`; The cell states. """
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
    """ Gated Recurrent Unit layer.\n
    - `x: Tensor`; 3d, with shapes `(Batch, Channel, Length)`.
    - `size: int`; Size of hidden features, and size of the output channel.
    - `init_weight_hh: None or str or callable`; Initialization specification for the hidden-hidden weight tensor.
        If a `str`, should be one of the nonlinearity functions contained in `torch.nn.init`.
        If a `callable`, it will be applied to `x` directly, i.e. `spec(x)`. If a 2-`tuple`,
        it must be of format `(callable, kwargs)`, i.e. `callable(x, **kwargs)`.
        Default: `'orthogonal_'`.
    - `init_weight_ih: None or str or callable`; Initialization specification for the input-hidden weight tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.GRU`s default scheme.
    - `init_bias_hh: None or str or callable`; Initialization specification for the hidden-hidden bias tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.GRU`s default scheme.
    - `init_bias_ih: None or str or callable`; Initialization specification for the input-hidden bias tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.GRU`s default scheme.
    - `bias: bool`; If `False`, then the layer does not use `bias_ih` and `bias_hh`. Default: `True`.
    - `num_layers: int`; Number of the recurrent layers. Default: 1.
    - `tuple_out: bool`; If `True`, the returned value will be a tuple `(out, (h_n, c_n))`. Default: False.
    - `**kw: dict`; Any additional KWargs are passed down to `torch.nn.GRU`, as well as `warm.engine.forward`.
        Refer to their docs for details. Some of the additional GRU arguments: `dropout, bidirectional, batch_first`.
    - `return: Tensor or tuple`; If `tuple_out` set to true, will return `(out, (h_n, c_n)`, otherwise just `out`.
        `out` has shape `(Batch, Size, Length*Directions)`, where `Directions` = 2 if `bidirectional` else 1.
        `h_n` has shape `(num_layers*Directions, Batch, Size)`; The hidden states. 
        `c_n` has shape `(num_layers*Directions, Batch, Size)`; The cell states. """
    return lstm(*arg, base_name='gru', base_class=nn.GRU, **kw)


def identity(x, *arg, **kw):
    """ Identity layer that returns the first input, ignores the rest arguments. """
    return x
