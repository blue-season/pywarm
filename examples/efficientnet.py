
# 09-20-2019;
"""
EfficientNet
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import warm
import warm.util
import warm.functional as W
from warm.engine import namespace


def swish(x):
    return x*torch.sigmoid(x)


def conv_pad_same(x, size, kernel=1, stride=1, **kw):
    pad = 0
    if kernel != 1 or stride != 1:
        in_size, s, k = [torch.as_tensor(v) for v in (x.shape[2:], stride, kernel)]
        pad = torch.max(((in_size+s-1)//s-1)*s+k-in_size, torch.tensor(0))
        left, right = pad//2, pad-pad//2
        if torch.all(left == right):
            pad = tuple(left.tolist())
        else:
            left, right = left.tolist(), right.tolist()
            pad = sum(zip(left[::-1], right[::-1]), ())
            x = F.pad(x, pad)
            pad = 0
    return W.conv(x, size, kernel, stride=stride, padding=pad, **kw)


@namespace
def conv_bn_act(x, size, kernel=1, stride=1, groups=1, bias=False, eps=1e-3, momentum=1e-2, act=swish, name='', **kw):
    x = conv_pad_same(x, size, kernel, stride=stride, groups=groups, bias=bias, name=name+'-conv')
    return W.batch_norm(x, eps=eps, momentum=momentum, activation=act, name=name+'-bn')


@namespace
def mb_block(x, size_out, expand=1, kernel=1, stride=1, se_ratio=0.25, dc_ratio=0.2, **kw):
    """ MobileNet Bottleneck Block. """
    size_in = x.shape[1]
    size_mid = size_in*expand
    y = conv_bn_act(x, size_mid, 1, **kw) if expand > 1 else x
    y = conv_bn_act(y, size_mid, kernel, stride=stride, groups=size_mid, **kw)
    y = squeeze_excitation(y, int(size_in*se_ratio), **kw)
    y = conv_bn_act(y, size_out, 1, act=None, **kw)
    if stride == 1 and size_in == size_out:
        y = drop_connect(y, dc_ratio)
        y += x
    return y


@namespace
def squeeze_excitation(x, size_se, name='', **kw):
    if size_se == 0:
        return x
    size_in = x.shape[1]
    x = F.adaptive_avg_pool2d(x, 1)
    x = W.conv(x, size_se, 1, activation=swish, name=name+'-conv1')
    return W.conv(x, size_in, 1, activation=swish, name=name+'-conv2')


def drop_connect(x, rate):
    """ Randomly set entire batch to 0. """
    if rate == 0:
        return x
    rate = 1.0-rate
    drop_mask = torch.rand([x.shape[0], 1, 1, 1], device=x.device, requires_grad=False)+rate
    return x/rate*drop_mask.floor()


spec_b0 = (
    (16, 1, 3, 1, 1, 0.25, 0.2), # size, expand, kernel, stride, repeat, se_ratio, dc_ratio
    (24, 6, 3, 2, 2, 0.25, 0.2),
    (40, 6, 5, 2, 2, 0.25, 0.2),
    (80, 6, 3, 2, 3, 0.25, 0.2),
    (112, 6, 5, 1, 3, 0.25, 0.2),
    (192, 6, 5, 2, 4, 0.25, 0.2),
    (320, 6, 3, 1, 1, 0.25, 0.2), )


class WarmEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        warm.up(self, [2, 3, 32, 32])
    def forward(self, x):
        x = conv_bn_act(x, 32, kernel=3, stride=2, name='head')
        for size, expand, kernel, stride, repeat, se_ratio, dc_ratio in spec_b0:
            for i in range(repeat):
                stride = stride if i == 0 else 1
                x = mb_block(x, size, expand, kernel, stride, se_ratio, dc_ratio)
        x = conv_bn_act(x, 1280, name='tail')
        x = F.adaptive_avg_pool2d(x, 1)
        x = W.dropout(x, 0.2)
        x = x.view(x.shape[0], -1)
        x = W.linear(x, 1000)
        return x


if __name__ == '__main__':
    m = WarmEfficientNet()
    warm.util.summary(m)
