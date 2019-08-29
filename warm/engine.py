# -*- coding: utf-8 -*-
# blue-season; 08-26-2019;
"""
"""
import torch
import torch.nn as nn
import numpy as np
import .util


_default_parent_module = None


def set_default_parent(parent):
    global _default_parent_module
    _default_parent_module = parent


def get_default_parent():
    global _default_parent_module
    return _default_parent_module


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


def prepare_model_(model, input=None, shape=None):
    with torch.no_grad():
        is_training = model.training
        model.eval()
        model(x)
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


def permute(x, in_shape='NCE', out_shape='NCE'):
    """ """
    if out_shape is None:
        return x
    if isinstance(in_shape, str) and isinstance(out_shape, str) :
        assert set(in_shape) == set(out_shape) == {'N', 'C', 'E'}, 'Shape can only have chars N, C, and E.'
        if in_shape == out_shape:
            return x
        pos = {c:i for i, c in enumerate(in_shape)}
        out_shape = [pos[c] for c in out_shape]
    return x.permute(out_shape)


def _forward(
        base_class, base_name=None, name=None, base_arg=None, base_kw=None, parent=None, prep_fn=None, post_fn=None,
        in_shape=None, base_shape=None, out_shape=None, forward_arg=None, forward_kw=None, tuple_out=False,
        initialization=None, activation=None, **kw):
    """ """
    parent = parent or get_default_parent()
    if name is None:
        base_name = base_name or util.camel_to_snake(base._get_name())
        name = _auto_name(base_name, parent)
    if name not in parent._modules:
        if prep_fn is not None:
            base_arg, base_kw = prep_fn(base_arg, base_kw)
        base = base_class(*(base_arg or []), **(base_kw or {}))
        parent.add_module(name, base)
        if initialization is not None:
            s = parent.state_dict()
            for k, v in initialization.items():
                initialize_(s[k], v)
    permute(x, in_shape, base_shape)
    x, *_ = parent._modules[name](x, *(forward_arg or []), **(forward_kw or {}))
    permute(x, base_shape, out_shape)
    if post_fn is not None:
        x, *_ = post_fn(x, *_)
    x = activate(x, activation)
    if tuple_out:
        return x, _
    return x
