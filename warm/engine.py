# 08-26-2019;
"""
PyWarm engine to the functional interface.
"""
import torch
import torch.nn as nn
import numpy as np
from warm import util


_DEFAULT_PARENT_MODULE = None


def set_default_parent(parent):
    """ Set the default `parent` module. """
    global _DEFAULT_PARENT_MODULE
    _DEFAULT_PARENT_MODULE = parent


def get_default_parent():
    """ Get the default `parent` module. """
    global _DEFAULT_PARENT_MODULE
    return _DEFAULT_PARENT_MODULE


def _auto_name(name, parent):
    """ Track the count of reference to `name` from `parent`. """
    if not is_ready(parent):
        parent._pywarm_auto_name_dict = {}
        def _hook(model, x):
            model._pywarm_auto_name_dict = {}
        parent._pywarm_forward_pre_hook = parent.register_forward_pre_hook(_hook)
    track = parent._pywarm_auto_name_dict
    if name not in track:
        track[name] = 0
    track[name] += 1
    return f'{name}_{track[name]}'


def prepare_model_(model, *data, device='cpu'):
    """ Initialize all childen modules defined by `warm` in a parent `model`.\n
    - `model: Module`; The parent model to be prepared.
    - `data: Tensor, or list of int`; A batch of data with the correct shape and type to be forwarded by model.
        `data` can also be a list of `int`, in which case it is interpreted as the shape of the input data.
    - `device: str, or torch.device`; Should be the same for `model` and `data`. Default: `'cpu'`.
    - `return: Module`; The prepared model, with all children modules defined by `warm` initialized. """
    _auto_name('', model)
    set_default_parent(model)
    def _prep_data(d):
        if isinstance(d, (np.ndarray, torch.Tensor)):
            return torch.as_tensor(d).to(device)
        elif isinstance(d, (list, tuple)):
            if all(isinstance(x, int) for x in d):
                return torch.randn(*d, device=device)
            return [_prep_data(x) for x in d]
        elif isinstance(d, dict):
            return {_prep_data(v) for k, v in d.items()}
    with torch.no_grad():
        is_training = model.training
        data = [_prep_data(d) for d in data]
        model.eval()
        model.to(device)
        model(*data)
        model.train(is_training)
    return model


def is_ready(model):
    """ Check if a `model` is prepared. """
    return hasattr(model, '_pywarm_forward_pre_hook')


def activate(x, spec, lookup=None):
    """ Activate tensors with given nonlinearity `spec`ification.\n
    - `x: Tensor or list of Tensor`; The tensors to be initialized.
    - `spec: str or callable or 2-tuple`; If a `str`, should be one of the nonlinearity functions contained in
        `torch.nn.functional` or `torch`. If a `callable`, it will be applied to `x` directly, i.e. `spec(x)`.
        If a 2-`tuple`, it must be of format `(callable, kwargs)`, i.e. `callable(x, **kwargs)`.
    - `lookup: None or list of module`; Parent modules to look for `spec`. If `None`, `[nn.functional, torch]` is used.
    - `return: Tensor or list of Tensor`; Activation results. """
    if spec is None:
        return x
    lookup = lookup or [nn.functional, torch]
    if isinstance(spec, str):
        for look in lookup:
            try:
                spec = getattr(look, spec)
                break
            except:
                pass
        if isinstance(spec, str):
            raise ValueError(f'Unknown spec {spec}.')
    if callable(spec):
        spec = (spec, {})
    fn, kw = spec
    if isinstance(x, (list, tuple)):
        return [fn(y, **kw) for y in x]
    return fn(x, **kw)


def initialize_(x, spec):
    """ Initialize parameters with given nonlinearity `spec`ification.\n
    - `x: Tensor or list of Tensor`; The tensors to be initialized.
    - `spec: str or callable or 2-tuple`; If a `str`, should be one of the nonlinearity functions contained in
        `torch.nn.init`. If a `callable`, it will be applied to `x` directly, i.e. `spec(x)`. If a 2-`tuple`,
        it must be of format `(callable, kwargs)`, i.e. `callable(x, **kwargs)`. """
    activate(x, spec, lookup=[nn.init])


def permute(x, in_shape='BCD', out_shape='BCD', **kw):
    """ Permute the dimensions of a tensor.\n
    - `x: Tensor`; The nd-tensor to be permuted.
    - `in_shape: str`; The dimension shape of `x`. Can only have characters `'B'` or `'C'` or `'D'`,
        which stand for Batch, Channel, or extra Dimensions. The default value `'BCD'` means
        the input tensor `x` should be at lest 2-d with shape `(Batch, Channel, Dim0, Dim1, Dim2, ...)`,
        where `Dim0, Dim1, Dim2 ...` stand for any number of extra dimensions.
    - `out_shape: str or tuple or None`; The dimension shape of returned tensor.  Default: `'BCD'`.
        If a `str`, it is restricted to the same three characters `'B'`, `'C'` or `'D'` as the `in_shape`.
        If a `tuple`, `in_shape` is ignored, and simply `x.permute(out_shape)` is returned.
        If `None`, no permution will be performed.
    - `return: Tensor`; Permuted nd-tensor. """
    if (in_shape == out_shape) or (out_shape is None):
        return x
    if isinstance(out_shape, (list, tuple, torch.Size)):
        return x.permute(*out_shape)
    if isinstance(in_shape, str) and isinstance(out_shape, str) :
        assert set(in_shape) == set(out_shape) <= {'B', 'C', 'D'}, 'In and out shapes must have save set of chars among B, C, and D.'
        in_shape = in_shape.lower().replace('d', '...')
        out_shape = out_shape.lower().replace('d', '...')
        return torch.einsum(f'{in_shape}->{out_shape}', x)
    return x


def unused_kwargs(kw):
    """ Filter out entries used by `forward` and return the rest. """
    fn_kw = dict(base_class=None,
        base_name=None, name=None, base_arg=None, base_kw=None, parent=None,
        infer_kw=None, in_shape='BCD', base_shape=None, out_shape='BCD', tuple_out=False,
        forward_arg=None, forward_kw=None, initialization=None, activation=None, )
    return {k:v for k, v in kw.items() if k not in fn_kw}


def forward(x, base_class, 
        base_name=None, name=None, base_arg=None, base_kw=None, parent=None,
        infer_kw=None, in_shape='BCD', base_shape='BCD', out_shape='BCD', tuple_out=False,
        forward_arg=None, forward_kw=None, initialization=None, activation=None, **kw):
    """ A forward template that creates child instances at the first time it is called.\n
    - `x: Tensor`; The nd-tensor to be forwarded.
    - `base_class: Module`; A child `torch.nn.Module` that will be created at the first time this function is called.
    - `base_name: str`; Name for the `base_class`. Default: base_class name.
    - `name: str`; Name for the child module instance. Default: class name plus ordinal.
    - `base_arg: tuple`; Positional args to be passed to create the child module instance. Default: None.
    - `base_kw: dict`; KWargs to be passed to create the child module instance. Default: None.
    - `parent: Module`; The parent of the child instance.  Default: None. If `None`, will use `get_default_parent`.
    - `infer_kw: dict`; Key should be valid for the child instance. Value shoud be a character,
        one of `'B'`, `'C'`, or `'D'` (see `permute`), to substitute for a dimension of `x`. Default: None.
    - `in_shape: str`; The dimension shape of `x`. See also `permute`. Default: `'BCD'`.
    - `base_shape: str`; The dimension shape required by the child module. See also `permute`. Default: `'BCD'`.
    - `out_shape: str or tuple or None`; The dimension shape of returned tensor. See also `permute`. Default: `'BCD'`.
    - `tuple_out: bool`; Whether the child module will return more than 1 outputs (e.g. `nn.RNN`).
        If `True`, the returned value of the function will be a tuple containing all outputs. Default: False.
    - `forward_arg: tuple`; positional args to be passed when calling the child module instance. Default: None.
    - `forward_kw: dict`; KWargs to be passed when calling the child module instance. Default: None.
    - `initialization: dict`; Keys are name of parameters to initialize. Values are init specs, which can be 
        a, `str`, a `callable`, or `2-tuple`; See the `spec` argument of `initialize_` for details. Default: None.
    - `activation: str or callable or 2-tuple`; See the `spec` argument of `activate`. Default: None.
    - `return: Tensor or tuple`; If `tuple_out` is `True`, the returned value will be a `tuple`. """
    parent = parent or get_default_parent()
    if name is None:
        base_name = base_name or util.camel_to_snake(base_class.__name__)
        name = _auto_name(base_name, parent)
    if name not in parent._modules:
        if infer_kw is not None:
            shape = in_shape
            if 'D' in shape:
                shape = list(shape)
                shape[shape.index('D')] = 'D'*(x.ndim-len(shape)+1)
                shape = ''.join(shape)
            infer_kw = {
                k:x.shape[shape.find(v) if isinstance(v, str) else v]
                for k, v in infer_kw.items()}
        base = base_class(*(base_arg or []), **(infer_kw or {}), **(base_kw or {}), )
        parent.add_module(name, base)
        if initialization is not None:
            s = parent.state_dict()
            for k, v in initialization.items():
                initialize_(s[name+'.'+k], v)
    x = permute(x, in_shape, base_shape)
    y = parent._modules[name](x, *(forward_arg or []), **(forward_kw or {}))
    r = []
    if isinstance(y, tuple):
        y, *r = y
    y = permute(y, base_shape, out_shape)
    y = activate(y, activation)
    if tuple_out:
        return (y, *r)
    return y
