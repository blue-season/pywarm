# -*- coding: utf-8 -*-
# 08-27-2019;
"""
"""
import torch
import torch.nn as nn
import numpy as np
from . import engine


class Lambda(nn.Module):
    def __init__(self, fn, *arg, **kw):
        super().__init__()
        self.fn = fn
        self.arg = arg
        self.kw = kw
    def forward(self, x):
        return self.fn(x, *arg, **kw)


class Permute(nn.Module):
    def __init__(self, in_shape='BCD', out_shape='BCD')
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
    def forward(self, x):
        return engine.permute(x, self.in_shape, self.out_shape)


class Sequential(nn.Sequential):
    def forward(self, x):
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
    def __init__(self, projection=None):
        super().__init__()
        self.projection = projection or nn.Identity()
    def forward(self, x):
        return super().forward(x)+self.projection(x)
