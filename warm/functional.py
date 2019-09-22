# 08-27-2019;
"""
Wraps around various torch.nn Modules to fit into a functional interface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from warm import engine
from warm import util

permute = engine.permute


def conv(x, size, kernel, init_weight=None, init_bias=None, bias=True, **kw):
    """ Convolution layer.\n
    -  `x: Tensor`; With shape `(Batch, Channel, *)` where `*` Can be 1d or 2d or 3d.
        If 3d, shapes are `(Batch, Channel, Length)`.
        If 4d, shapes are `(Batch, Channel, Height, Width)`.
        If 5d, shapes are `(Batch, Channel, Depth, Height, Width)`.
    -  `size: int`; Size of hidden filters, and size of the output channel.
    -  `kernel: int or tuple`; Size of the convolution kernel.
    -  `init_weight: None or str or callable`; Initialization specification for the weight tensor.
        If a `str`, should be one of the nonlinearity functions contained in `torch.nn.init`.
        If a `callable`, it will be applied to `x` directly, i.e. `spec(x)`. If a 2-`tuple`,
        it must be of format `(callable, kwargs)`, i.e. `callable(x, **kwargs)`.
        Default: `None`, and the weight tensor is initialized using `torch.nn.ConvNd`s default scheme.
    -  `init_bias: None or str or callable`; Same as `init_weight`, but for the bias tensor.
    -  `bias: bool`; If `True`, adds a learnable bias to the output. Default: `True`.
    -  `**kw:dict`; Any additional KWargs are passed down to `torch.nn.ConvNd`, where N can be 1, 2 or 3.
        as well as `warm.engine.forward`. Refer to their docs for details. Some of the additional ConvNd arguments:
        `stride, padding, dilation, groups`.
    -  `return: Tensor`; With shape `(Batch, Size, *)` where `*` can be 1d, 2d, 3d that depends on `x`. """
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
    -  `x: Tensor`; 2d or more, with shapes `(Batch, Channel, *)` where `*` means any number of additional dimensions.
    -  `size: int`; Size of hidden features, and size of the output channel.
    -  `init_weight: None or str or callable`; Initialization specification for the weight tensor.
        If a `str`, should be one of the nonlinearity functions contained in `torch.nn.init`.
        If a `callable`, it will be applied to `x` directly, i.e. `spec(x)`. If a 2-`tuple`,
        it must be of format `(callable, kwargs)`, i.e. `callable(x, **kwargs)`.
        Default: `None`, and the weight tensor is initialized using `torch.nn.Linear`s default scheme.
    -  `init_bias: None or str or callable`; Same as `init_weight`, but for the bias tensor.
    -  `bias: bool`; If `True`, adds a learnable bias to the output. Default: `True`.
    -  `**kw:dict`; Any additional KWargs are passed down to `warm.engine.forward`. Refer to its docs for details.
    -  `return: Tensor`; With shape `(Batch, Size, *)` where `*` can be 1d, 2d, 3d that depends on `x`. """
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
    -  `x: Tensor`; 2d or more, with shapes `(Batch, Channel, *)` where `*` means any number of additional dimensions.
    -  `**kw: dict`; Any additional KWargs are passed down to `torch.nn.BatchNormNd`, where N can be 1, 2 or 3.
        as well as `warm.engine.forward`. Refer to their docs for details. Some of the additional BatchNorm arguments:
        `eps, momentum, affine, track_running_stats`.
    -  `return: Tensor`; Same shape as input  `x`. """
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
    -  `x: Tensor or tuple`; If tuple, must be of format `(x, (h_0, c_0))`, where `x` is a 3d tensor,
        with shapes `(Batch, Channel, Length)`.
    -  `size: int`; Size of hidden features, and size of the output channel.
    -  `init_weight_hh: None or str or callable`; Initialization specification for the hidden-hidden weight tensor.
        If a `str`, should be one of the nonlinearity functions contained in `torch.nn.init`.
        If a `callable`, it will be applied to `x` directly, i.e. `spec(x)`. If a 2-`tuple`,
        it must be of format `(callable, kwargs)`, i.e. `callable(x, **kwargs)`.
        Default: `'orthogonal_'`.
    -  `init_weight_ih: None or str or callable`; Initialization specification for the input-hidden weight tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.LSTM`s default scheme.
    -  `init_bias_hh: None or str or callable`; Initialization specification for the hidden-hidden bias tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.LSTM`s default scheme.
    -  `init_bias_ih: None or str or callable`; Initialization specification for the input-hidden bias tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.LSTM`s default scheme.
    -  `bias: bool`; If `False`, then the layer does not use `bias_ih` and `bias_hh`. Default: `True`.
    -  `num_layers: int`; Number of the recurrent layers. Default: 1.
    -  `tuple_out: bool`; If `True`, the returned value will be a tuple `(out, (h_n, c_n))`. Default: False.
    -  `**kw: dict`; Any additional KWargs are passed down to `torch.nn.LSTM`, as well as `warm.engine.forward`.
        Refer to their docs for details. Some of the additional LSTM arguments: `dropout, bidirectional, batch_first`.
    -  `return: Tensor or tuple`; If `tuple_out` set to true, will return `(out, (h_n, c_n)`, otherwise just `out`.
        `out` has shape `(Batch, Size, Length*Directions)`,
            where Directions = 2 if `bidirectional` else 1.
        `h_n` is the hidden states with shape `(num_layers*Directions, Batch, Size)`.
        `c_n` is the cell states with shape `(num_layers*Directions, Batch, Size)`. """
    states = None
    if isinstance(x, tuple):
        x, *states = x
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
        forward_arg=states,
        initialization={
            f'{k}_l{l}':init[k] for k in ['weight_hh', 'weight_ih']+(['bias_hh', 'bias_ih'] if bias else [])
            for l in range(num_layers)}, )
    return engine.forward(x, **{**inferred_kw, **kw})


def gru(*arg, **kw):
    """ Gated Recurrent Unit layer.\n
    -  `x: Tensor or tuple`; If tuple, must be of format `(x, (h_0, c_0))`, where `x` is a 3d tensor,
        with shapes `(Batch, Channel, Length)`.
    -  `size: int`; Size of hidden features, and size of the output channel.
    -  `init_weight_hh: None or str or callable`; Initialization specification for the hidden-hidden weight tensor.
        If a `str`, should be one of the nonlinearity functions contained in `torch.nn.init`.
        If a `callable`, it will be applied to `x` directly, i.e. `spec(x)`. If a 2-`tuple`,
        it must be of format `(callable, kwargs)`, i.e. `callable(x, **kwargs)`.
        Default: `'orthogonal_'`.
    -  `init_weight_ih: None or str or callable`; Initialization specification for the input-hidden weight tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.GRU`s default scheme.
    -  `init_bias_hh: None or str or callable`; Initialization specification for the hidden-hidden bias tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.GRU`s default scheme.
    -  `init_bias_ih: None or str or callable`; Initialization specification for the input-hidden bias tensor.
        Default: `None`, and the weight tensor is initialized using `torch.nn.GRU`s default scheme.
    -  `bias: bool`; If `False`, then the layer does not use `bias_ih` and `bias_hh`. Default: `True`.
    -  `num_layers: int`; Number of the recurrent layers. Default: 1.
    -  `tuple_out: bool`; If `True`, the returned value will be a tuple `(out, (h_n, c_n))`. Default: False.
    -  `**kw: dict`; Any additional KWargs are passed down to `torch.nn.GRU`, as well as `warm.engine.forward`.
        Refer to their docs for details. Some of the additional GRU arguments: `dropout, bidirectional, batch_first`.
    -  `return: Tensor or tuple`; If `tuple_out` set to true, will return `(out, (h_n, c_n)`, otherwise just `out`.
        `out` has shape `(Batch, Size, Length*Directions)`,
            where Directions = 2 if `bidirectional` else 1.
        `h_n` is the hidden states with shape `(num_layers*Directions, Batch, Size)`.
        `c_n` is the cell states with shape `(num_layers*Directions, Batch, Size)`. """
    return lstm(*arg, base_name='gru', base_class=nn.GRU, **kw)


def identity(x, *arg, **kw):
    """ Identity layer that returns the first input, ignores the rest arguments. """
    return x


def dropout(x, rate=0.5, by_channel=False, **kw):
    """ Dropout layer.\n
    During training, randomly zeros part of input tensor `x`, at probability `rate`.\n
    -  `x: Tensor`; Can be of any shape if `by_channel` is false, or 2d and up if `by_channel` is true.
    -  `rate: float`; The probability of dropout. Default 0.5.
    -  `by_channel: bool`; If true, will dropout entire channels (all `'D'` dimensions will be 0 if x is `'BCD'`).
        `by_channel` true requires `x` to be 2d or more.
    -  `inplace: bool`; If true, the operation will be in-place and the input `x` will be altered.
    -  `return: Tensor`; Same shape as `x`. """
    inferred_kw = dict(
        base_name='dropout',
        base_class=[nn.Dropout, nn.Dropout2d][by_channel],
        base_kw={'p':rate},
        base_shape=[None, 'BCD'][by_channel], )
    return engine.forward(x, **{**inferred_kw, **kw})


def transformer(x, y=None, num_encoder=6, num_decoder=6, num_head=8,
        mask=None, causal=False, in_shape='BCD', **kw):
    """ Transformer layer.\n
    This layer covers functionality of `Transformer`, `TransformerEncoder`, and `TransformerDecoder`.
    See [`torch.nn.Transformer`](https://pytorch.org/docs/stable/nn.html#transformer) for more details.\n
    -  `x: Tensor`; The source sequence, with shape `(Batch, Channel, LengthX)`.
        `Channel` is usually from embedding.
    -  `y: None or Tensor`; The target sequence. Also with shape `(Batch, Channel, LengthY)`.
        If not present, default to equal `x`.
    -  `num_encoder: int`; Number of encoder layers. Set to 0 to disable encoder and use only decoder. Default 6.
    -  `num_decoder: int`; Number of decoder layers. Set to 0 to disable decoder and use only encoder. Default 6.
    -  `num_head: int`; Number of heads for multi-headed attention. Default 8.
    -  `mask: None or dict`; Keys are among: `src_mask`, `tgt_mask`, `memory_mask`,
        `src_key_padding_mask`, `tgt_key_padding_mask`, `memory_key_padding_mask`.
        See the `forward` method of `torch.nn.Transformer` for details.
    -  `causal: bool`; Default false. if true, will add causal masks to source and target, so that
        current value only depends on the past, not the future, in the sequences.
    -  `**kw: dict`; Any additional KWargs are passed down to `torch.nn.Transformer`, as well as `warm.engine.forward`.
    -  `return: Tensor`; Same shape as `y`, if `num_decoder` > 0. Otherwise same shape as `x`. """
    def _causal_mask(n):
        mask = (torch.triu(torch.ones(n, n)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    if y is None:
        y = x
    y = permute(y, in_shape, 'DBC')
    mask = mask or {}
    if causal:
        i = in_shape.find('D')
        mx = _causal_mask(x.shape[i])
        mask['src_mask'] = mask.pop('src_mask', 0.0)+mx
        my = _causal_mask(y.shape[0])
        mask['tgt_mask'] = mask.pop('tgt_mask', 0.0)+my
    encoder = identity if num_encoder == 0 else None
    decoder = identity if num_decoder == 0 else None
    inferred_kw = dict(
        base_name='transformer',
        base_class=nn.Transformer,
        base_shape='DBC',
        base_kw=dict(
            d_model=x.shape[in_shape.find('C')],
            custom_encoder=encoder,
            custom_decoder=decoder,
            nhead=num_head,
            num_encoder_layers=num_encoder,
            num_decoder_layers=num_decoder, 
            **engine.unused_kwargs(kw), ),
        in_shape=in_shape,
        forward_kw=mask,
        forward_arg=(y, ), )
    return engine.forward(x, **{**inferred_kw, **kw})


def layer_norm(x, dim=1, **kw):
    """ Layer Normalization.\n
    -  `x: Tensor`; Can be of any shape.
    -  `dim: int or list of int`; Dimensions to be normalized. Default: 1.
    -  `**kw: dict`; Any additional KWargs are passed down to `torch.nn.LayerNorm`, as well as `warm.engine.forward`.
    -  `return: Tensor`; Same shape as `x`. """
    if dim != -1:
        if isinstance(dim, int):
            dim = [dim]
        dim_norm = [x.ndim+i if i < 0 else i for i in dim]
        order = [i for i in range(x.ndim) if i not in dim_norm]+dim_norm
        x = x.permute(order)
        norm_shape = x.shape[-len(dim_norm):]
    else:
        norm_shape = [x.shape[-1]]
    inferred_kw = dict(
        base_name='layer_norm',
        base_class=nn.LayerNorm,
        base_kw={'normalized_shape':norm_shape}, )
    x = engine.forward(x, **{**inferred_kw, **kw})
    if dim != -1:
        x = x.permute(np.argsort(order).tolist())
    return x


def embedding(x, size, vocabulary=None, **kw):
    """ Embedding layer.\n
    The input is usually a list of indices (integers), and the output is a dense matrix which
    maps indices to dense vectors. Thus the output will have 1 more dimension than the input.\n
    **Note**: The output of this function is always one more dimension than the input. For input with shape `(*)`,
    The output will be `(*, size)`. Any shape specifications in the KWargs are ignored. \n
    -  `x: Tensor`; Contains indices into the vocabulary. Will be converted to `LongTensor` of integers.
        Can be of any shape.
    -  `size: int`; The size of embedding vector.
    -  `vocabulary: int or None`; The size of vocabulary of embedding, or max number of unique indices in `x`.
        By default it is set to `max(x)-min(x)+1`.
    -  `**kw: dict`; Any additional KWargs are passed down to `torch.nn.LayerNorm`, as well as `warm.engine.forward`.
    -  `return: Tensor`; With the embedded dim appended to the shape of x.
        Thus with shape `(*, Size)`, where `*` is the shape of `x`. """
    x = x.type(torch.LongTensor)
    if vocabulary is None:
        vocabulary = x.max()-x.min()+1
    kw.pop('in_shape', None)
    kw.pop('out_shape', None)
    kw.pop('base_shape', None)
    inferred_kw = dict(
        base_name='embedding',
        base_class=nn.Embedding,
        base_kw=dict(
            num_embeddings=vocabulary,
            embedding_dim=size,
            **engine.unused_kwargs(kw), ),
        base_shape=None,
        in_shape=None,
        out_shape=None, )
    return engine.forward(x, **{**inferred_kw, **kw})
