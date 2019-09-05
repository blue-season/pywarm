# 08-27-2019;
"""
Custom modules to enhance the nn Sequential experience.

PyWarm's core concept is to use a functional interface to simplify network building.
However, if you still prefer the classical way of defining child modules in `__init__()`,
Pywarm provides some utility modules to help organize child modules better.

`Lambda` can be used to wrap one line data transformations, like `x.view()`, `x.permute()` etc, into modules.

`Sequential` is an extension to `nn.Sequential` that better accomodates PyTorch RNNs.

`Shortcut` is another extension to `nn.Sequential` that will also perform a shortcut addition (AKA residual connection)
for the input with output, so that residual blocks can be written in an entire sequential way.

For example, to define the basic block type for resnet:


```Python
import torch.nn as nn
import warm.module as wm


def basic_block(size_in, size_out, stride=1):
    block = wm.Shortcut(
        nn.Conv2d(size_in, size_out, 3, stride, 1, bias=False),
        nn.BatchNorm2d(size_out),
        nn.ReLU(),
        nn.Conv2d(size_out, size_out, 3, 1, 1, bias=False),
        nn.BatchNorm2d(size_out),
        projection=wm.Lambda(
            lambda x: x if x.shape[1] == size_out else nn.Sequential(
                nn.Conv2d(size_in, size_out, 1, stride, bias=False),
                nn.BatchNorm2d(size_out), )(x), ), )
    return block
```
"""


import torch.nn as nn


class Lambda(nn.Module):
    """ Wraps a callable and all its call arguments.\n
    - `fn: callable`; The callable being wrapped.
    - `*arg: list`; Arguments to be passed to `fn`.
    - `**kw: dict`; KWargs to be passed to `fn`. """
    def __init__(self, fn, *arg, **kw):
        super().__init__()
        self.fn = fn
        self.arg = arg
        self.kw = kw
    def forward(self, x):
        """ """
        return self.fn(x, *self.arg, **self.kw)


class Sequential(nn.Sequential):
    """ Similar to `nn.Sequential`, except that child modules can have multiple outputs (e.g. `nn.RNN`).\n
    - `*arg: list of Modules`; Same as `nn.Sequential`. """
    def forward(self, x):
        """ """
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
        """ """
        return super().forward(x)+self.projection(x)
