# 09-03-2019;
"""
Construct a WarmMobileNetV2() using PyWarm, then copy state dicts
from torchvision.models.mobilenet_v2() into WarmMobileNetV2(),
compare if it produce identical results as the official one.
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


def conv_bn_relu(x, size, stride=1, expand=1, kernel=3, groups=1, name=''):
    x = W.conv(x, size, kernel, padding=(kernel-1)//2, stride=stride, groups=groups, bias=False,
        name=f'{name}-0', )
    return W.batch_norm(x, activation='relu6', name=f'{name}-1')


def bottleneck(x, size_out, stride, expand, name=''):
    size_in = x.shape[1]
    size_mid = size_in*expand
    y = conv_bn_relu(x, size_mid, kernel=1, name=f'{name}-conv-0') if expand > 1 else x
    y = conv_bn_relu(y, size_mid, stride, kernel=3, groups=size_mid, name=f'{name}-conv-{1 if expand > 1 else 0}')
    y = W.conv(y, size_out, kernel=1, bias=False, name=f'{name}-conv-{2 if expand > 1 else 1}')
    y = W.batch_norm(y, name=f'{name}-conv-{3 if expand > 1 else 2}')
    if stride == 1 and size_in == size_out:
        y += x # residual shortcut
    return y


def conv1x1(x, *arg, **kw):
    return conv_bn_relu(x, *arg, kernel=1, **kw)


def pool(x, *arg, **kw):
    return x.mean([2, 3])


def classify(x, size, *arg, **kw):
    x = W.dropout(x, rate=0.2, name='classifier-0')
    return W.linear(x, size, name='classifier-1')


default_spec = (
    (None, 32, 1, 2, conv_bn_relu),  # t, c, n, s, operator
    (1, 16, 1, 1, bottleneck),
    (6, 24, 2, 2, bottleneck),
    (6, 32, 3, 2, bottleneck),
    (6, 64, 4, 2, bottleneck),
    (6, 96, 3, 1, bottleneck),
    (6, 160, 3, 2, bottleneck),
    (6, 320, 1, 1, bottleneck),
    (None, 1280, 1, 1, conv1x1),
    (None, None, 1, None, pool),
    (None, 1000, 1, None, classify), )


class WarmMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        warm.engine.prepare_model_(self, [2, 3, 224, 224])
    def forward(self, x):
        count = 0
        for t, c, n, s, op in default_spec:
            for i in range(n):
                stride = s if i == 0 else 1
                x = op(x, c, stride, t, name=f'features-{count}')
                count += 1
        return x


def test():
    """ Compare the classification result of WarmMobileNetV2 versus torchvision mobilenet_v2. """
    new = WarmMobileNetV2()
    from torchvision.models import mobilenet_v2
    old = mobilenet_v2()
    state = old.state_dict()
    for k in list(state.keys()): # Map parameters of old, e.g. layer2.0.conv1.weight
        s = k.split('.') # to parameters of new, e.g. layer2-0-conv1.weight
        s = '-'.join(s[:-1])+'.'+s[-1]
        state[s] = state.pop(k)
    new.load_state_dict(state)
    warm.util.summary(old)
    warm.util.summary(new)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        old.eval()
        y_old = old(x)
        new.eval()
        y_new = new(x)
        if torch.equal(y_old, y_new):
            print('Success! Same results from old and new.')
        else:
            print('Warning! New and old produce different results.')


if __name__ == '__main__':
    test()
