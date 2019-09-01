# -*- coding: utf-8 -*-
# 08-27-2019;
"""
"""
import torch
import torch.nn as nn
import numpy as np
from . import engine


class Lambda(nn.Module):
    """ Wraps a callable and all its call arguments. """
    def __init__(self, fn, *arg, **kw):
        super().__init__()
        self.fn = fn
        self.arg = arg
        self.kw = kw
    def forward(self, x):
        """ Forward will be perform at every call. """
        return self.fn(x, *self.arg, **self.kw)


class Sequential(nn.Sequential):
    """ Similar to `nn.Sequential`, except that child modules can have multiple outputs (e.g. `nn.RNN`). """
    def forward(self, x):
        """ Forward will be perform at every call. """
        for module in self._modules.values():
            if isinstance(x, tuple):
                try:
                    x = module(x)
                except Exception:
                    x = module(x[0])
            else:
                x = module(x)
        return x


class Shortcut(Sequential):
    """ Similar to `nn.Sequential`, except that it performs a shortcut addition for the input and output.\n
    - `*arg: list of Modules`; Same as `nn.Sequential`.
    - `projection: None or callable`; If `None`, input with be added directly to the output.
        otherwise input will be passed to the `projection` first, usually to make the shapes match. """
    def __init__(self, *arg, projection=None):
        super().__init__(*arg)
        self.projection = projection or nn.Identity()
    def forward(self, x):
        return super().forward(x)+self.projection(x)
