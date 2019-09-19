# 08-29-2019;
"""
Construct a WarmResNet() using PyWarm, then copy state dicts
from torchvision.models.resnet18() into WarmResNet(),
compare if it produce identical results as the official one.
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append('..')
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import warm
import warm.util
import warm.functional as W


def basic(x, size, stride, stack_index, block_index):
    """ The basic block. """
    prefix = f'layer{stack_index+1}-{block_index}-'
    y = W.conv(x, size, 3, stride=stride, padding=1, bias=False, name=prefix+'conv1')
    y = W.batch_norm(y, activation='relu', name=prefix+'bn1')
    y = W.conv(y, size, 3, stride=1, padding=1, bias=False, name=prefix+'conv2')
    y = W.batch_norm(y, name=prefix+'bn2')
    if y.shape[1] != x.shape[1]:
        x = W.conv(x, y.shape[1], 1, stride=stride, bias=False, name=prefix+'downsample-0')
        x = W.batch_norm(x, name=prefix+'downsample-1')
    return F.relu(y+x)


def stack(x, num_block, size, stride, stack_index, block=basic):
    """ A stack of num_block blocks. """
    for block_index, s in enumerate([stride]+[1]*(num_block-1)):
        x = block(x, size, s, stack_index, block_index)
    return x


class WarmResNet(nn.Module):
    def __init__(self, block=basic, stack_spec=((2, 64, 1), (2, 128, 2), (2, 256, 2), (2, 512, 2))):
        super().__init__()
        self.block = block
        self.stack_spec = stack_spec
        warm.up(self, [2, 3, 32, 32])
    def forward(self, x):
        y = W.conv(x, 64, 7, stride=2, padding=3, bias=False, name='conv1')
        y = W.batch_norm(y, activation='relu', name='bn1')
        y = F.max_pool2d(y, 3, stride=2, padding=1)
        for i, spec in enumerate(self.stack_spec):
            y = stack(y, *spec, i, block=self.block)
        y = F.adaptive_avg_pool2d(y, 1)
        y = torch.flatten(y, 1)
        y = W.linear(y, 1000, name='fc')
        return y


def test_time(fn, *arg, repeat=10, **kw):
    dur = 0.0
    for i in range(repeat):
        start = time.time()
        y = fn(*arg, **kw)
        dur += time.time()-start
    return dur


def test():
    """ Compare the classification result of WarmResNet versus torchvision resnet18. """
    new = WarmResNet()
    from torchvision.models import resnet18
    old = resnet18()
    state = old.state_dict()
    for k in list(state.keys()): # Map parameters of old, e.g. layer2.0.conv1.weight
        s = k.split('.') # to parameters of new, e.g. layer2-0-conv1.weight
        s = '-'.join(s[:-1])+'.'+s[-1]
        state[s] = state.pop(k)
    new.load_state_dict(state)
    warm.util.summary(old)
    warm.util.summary(new)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        old.eval()
        y_old = old(x)
        new.eval()
        y_new = new(x)
        if torch.equal(y_old, y_new):
            print('Success! Same results from old and new.')
        else:
            print('Warning! New and old produce different results.')
        t_old = test_time(old, x)
        t_new = test_time(new, x)
        print('Total forward time for old:', t_old, 'seconds.')
        print('Total forward time for new:', t_new, 'seconds.')


if __name__ == '__main__':
    test()
