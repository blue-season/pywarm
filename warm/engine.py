# -*- coding: utf-8 -*-
# 08-26-2019;
"""
"""
import torch
import torch.nn as nn
import numpy as np
from . import util


_DEFAULT_PARENT_MODULE = None


def set_default_parent(parent):
    global _DEFAULT_PARENT_MODULE
    _DEFAULT_PARENT_MODULE = parent


def get_default_parent():
    global _DEFAULT_PARENT_MODULE
    return _DEFAULT_PARENT_MODULE


def _auto_name(name, parent):
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


def prepare_model_(model, data, device='cpu'):
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
        data = _prep_data(data)
        model.eval()
        model.to(device)
        model(data)
        model.train(is_training)
    return model


def is_ready(model):
    return hasattr(model, '_pywarm_forward_pre_hook')


def initialize_(param, spec):
    """ """
    if spec is None:
        return
    if isinstance(spec, str):
        assert hasattr(nn.init, spec), f'Unknow initialization: {spec}.'
        spec = getattr(nn.init, spec)
    if callable(spec):
        spec = (spec, {})
    fn, kw = spec
    if not isinstance(param, (list, tuple)):
        param = [param]
    for p in param:
        fn(p, **kw)


def activate(x, spec):
    """ """
    if spec is None:
        return x
    if isinstance(spec, str):
        assert hasattr(nn.functional, spec), f'Unknow activation: {spec}.'
        spec = getattr(nn.functional, spec)
    if callable(spec):
        spec = (spec, {})
    fn, kw = spec
    return fn(x, **kw)


def permute(x, in_shape='BCD', out_shape='BCD'):
    """ """
    if (in_shape == out_shape) or (out_shape is None):
        return x
    if isinstance(out_shape, (list, tuple, torch.Size)):
        return x.permute(*out_shape)
    if isinstance(in_shape, str) and isinstance(out_shape, str) :
        assert set(in_shape) == set(out_shape) <= {'B', 'C', 'D'}, 'In and out shapes must have save set of chars among B, C, and D.'
        if x.ndim == 2:
            in_shape = in_shape.replace('D', '')
            out_shape = out_shape.replace('D', '')
        if x.ndim <= 3:
            return x.permute(*[in_shape.find(d) for d in out_shape])
        dim = {'B':1, 'C':1, 'D':x.ndim-2}
        dim = np.split(list(x.shape), np.cumsum([dim[d] for d in in_shape]))[:-1]
        dim = {d:v for d, v in zip(in_shape, dim)}
        dd = dim['D']
        dim = {'B':int(dim['B']), 'C':int(dim['C']), 'D':-1}
        x = torch.reshape(x, [dim[d] for d in in_shape])
        x = x.permute(*[in_shape.find(d) for d in out_shape])
        dim['D'] = dd
        x = torch.reshape(x, list(np.hstack([dim[d] for d in out_shape])))
    return x


def unused_kwargs(kw):
    fn_kw = dict(base_class=None,
        base_name=None, name=None, base_arg=None, base_kw=None, parent=None,
        infer_kw=None, in_shape='BCD', base_shape=None, out_shape='BCD', tuple_out=False,
        forward_arg=None, forward_kw=None, initialization=None, activation=None, )
    return {k:v for k, v in kw.items() if k not in fn_kw}


def forward(x, base_class, 
        base_name=None, name=None, base_arg=None, base_kw=None, parent=None,
        infer_kw=None, in_shape='BCD', base_shape='BCD', out_shape='BCD', tuple_out=False,
        forward_arg=None, forward_kw=None, initialization=None, activation=None, **kw):
    """ """
    parent = parent or get_default_parent()
    if name is None:
        base_name = base_name or util.camel_to_snake(base_class.__name__)
        name = _auto_name(base_name, parent)
    if name not in parent._modules:
        if infer_kw is not None:
            infer_kw = {
                k:x.shape[in_shape.find(v) if isinstance(v, str) else v]
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
