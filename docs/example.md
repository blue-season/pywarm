
# PyWarm Examples

## ResNet

A more detailed example, the ResNet18 network defined in PyWarm and vanilla PyTorch:

``` Python tab="Warm" linenums="1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import warm
import warm.functional as W


def basic(x, size, stride):
    y = W.conv(x, size, 3, stride=stride, padding=1, bias=False)
    y = W.batch_norm(y, activation='relu')
    y = W.conv(y, size, 3, stride=1, padding=1, bias=False)
    y = W.batch_norm(y)
    if y.shape[1] != x.shape[1]: # channel size mismatch, needs projection
        x = W.conv(x, y.shape[1], 1, stride=stride, bias=False)
        x = W.batch_norm(x)
    y = y+x # residual shortcut connection
    return F.relu(y)


def stack(x, num_block, size, stride, block=basic):
    for s in [stride]+[1]*(num_block-1):
        x = block(x, size, s)
    return x


class ResNet(nn.Module):

    def __init__(self, block=basic,
            stack_spec=((2, 64, 1), (2, 128, 2), (2, 256, 2), (2, 512, 2))):
        super().__init__()
        self.block = block
        self.stack_spec = stack_spec
        warm.engine.prepare_model_(self, [2, 3, 32, 32])

    def forward(self, x):
        y = W.conv(x, 64, 7, stride=2, padding=3, bias=False)
        y = W.batch_norm(y, activation='relu')
        y = F.max_pool2d(y, 3, stride=2, padding=1)
        for spec in self.stack_spec:
            y = stack(y, *spec, block=self.block)
        y = F.adaptive_avg_pool2d(y, 1)
        y = torch.flatten(y, 1)
        y = W.linear(y, 1000)
        return y


resnet18 = ResNet()
```

``` Python tab="Torch" linenums="1"
# code based on torchvision/models/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(size_in, size_out, stride=1):
    return nn.Conv2d(size_in, size_out, kernel_size=3, stride=stride,
        padding=1, groups=1, bias=False, dilation=1, )


def conv1x1(size_in, size_out, stride=1):
    return nn.Conv2d(size_in, size_out, kernel_size=1, stride=stride,
        padding=0, groups=1, bias=False, dilation=1, )


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, size_in, size_out, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(size_in, size_out, stride)
        self.bn1 = nn.BatchNorm2d(size_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(size_out, size_out)
        self.bn2 = nn.BatchNorm2d(size_out)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.downsample is not None:
            identity = self.downsample(x)
        y += identity
        y = self.relu(y)
        return y


class ResNet(nn.Module):

    def __init__(self,
            block=BasicBlock, num_block=[2, 2, 2, 2]):
        super().__init__()
        self.size_in = 64
        self.conv1 = nn.Conv2d(3, self.size_in, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.size_in)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stack1 = self._make_stack(block, 64, num_block[0], 1)
        self.stack2 = self._make_stack(block, 128, num_block[1], 2)
        self.stack3 = self._make_stack(block, 256, num_block[2], 2)
        self.stack4 = self._make_stack(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1000)

    def _make_stack(self, block, size_out, num_blocks, stride):
        downsample = None
        if stride != 1 or self.size_in != size_out:
            downsample = nn.Sequential(
                conv1x1(self.size_in, size_out, stride),
                nn.BatchNorm2d(size_out), )
        stacks = []
        for stride in strides:
            stacks.append(
                block(self.size_in, size_out, stride, downsample))
            self.size_in = size_out
        return nn.Sequential(*stacks)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.stack1(y)
        y = self.stack2(y)
        y = self.stack3(y)
        y = self.stack4(y)
        y = self.avg_pool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return y


resnet18 = ResNet()
```

-   The PyWarm version significantly reduces self-repititions of code as in the vanilla PyTorch version.

-   Note that when warming the model via `warm.engine.prepare_model_(self, [2, 3, 32, 32])`
    We set the first `Batch` dimension to 2 because the model uses `batch_norm`,
    which will not work when `Batch` is 1.

----

## MobileNet

``` Python tab="Warm" linenums="1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import warm
import warm.functional as W


def conv_bn_relu(x, size, stride=1, expand=1, kernel=3, groups=1):
    x = W.conv(x, size, kernel, padding=(kernel-1)//2,
        stride=stride, groups=groups, bias=False, )
    return W.batch_norm(x, activation='relu6')


def bottleneck(x, size_out, stride, expand):
    size_in = x.shape[1]
    size_mid = size_in*expand
    y = conv_bn_relu(x, size_mid, kernel=1) if expand > 1 else x
    y = conv_bn_relu(y, size_mid, stride, kernel=3, groups=size_mid)
    y = W.conv(y, size_out, kernel=1, bias=False)
    y = W.batch_norm(y)
    if stride == 1 and size_in == size_out:
        y += x # residual shortcut
    return y


def conv1x1(x, *arg):
    return conv_bn_relu(x, *arg, kernel=1)


def pool(x, *arg):
    return x.mean([2, 3])


def classify(x, size, *arg):
    x = W.dropout(x, rate=0.2)
    return W.linear(x, size)


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


class MobileNetV2(nn.Module):

    def __init__(self):
        super().__init__()
        warm.engine.prepare_model_(self, [2, 3, 224, 224])
        
    def forward(self, x):
        for t, c, n, s, op in default_spec:
            for i in range(n):
                stride = s if i == 0 else 1
                x = op(x, c, stride, t)
        return x


net = MobileNetV2()
```

``` Python tab="Torch" linenums="1"
# code based on torchvision/models/mobilenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, 
            kernel_size=3, stride=1, groups=1):
        padding = (kernel_size-1)//2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, 
                stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True), )


class BottleNeck(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, 
                stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup), ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


default_spec = [
    [1, 16, 1, 1], # t, c, n, s
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1], ]


class MobileNetV2(nn.Module):

    def __init__(self):
        super().__init__()
        input_channel = 32
        last_channel = 1280
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in default_spec:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    BottleNeck(
                        input_channel, output_channel,
                        stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, 
            last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, 1000), )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


net = MobileNetV2()
```

## Transformer

```Python
"""
The Transformer model from paper Attention is all you need.
The Transformer instance accepts two inputs:
x is Tensor with shape (Batch, Channel, LengthX).
    usually a source sequence from embedding (in such cases,
    Channel equals the embedding size).
y is Tensor with shape (Batch, Channel, lengthY).
    usually a target sequence, also from embedding.
**kw is passed down to inner components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import warm
import warm.functional as W


def multi_head_attention(x, y=None, num_head=8, dropout=0.1, mask=None, **kw):
    def split_heads(t):
        return t.reshape(batch, num_head, size//num_head, t.shape[-1])
    def merge_heads(t):
        return t.reshape(batch, -1, t.shape[-1])
    if y is None:
        y = x # self attention
    batch, size = x.shape[:2]
    assert size%num_head == 0, 'num_head must be a divisor of size.'
    assert y.shape[:2] == x.shape[:2], 'The first 2 dims of x, y must match.'
    q = W.linear(x, size) # query
    k = W.linear(y, size) # key
    v = W.linear(y, size) # value
    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)
    q *= (size//num_head)**(-0.5)
    a = q.transpose(2, 3).contiguous().matmul(k) # attention weights
    if mask is not None:
        a += mask
    a = F.softmax(a, dim=-1)
    a = W.dropout(a, dropout)
    x = v.matmul(a.transpose(2, 3).contiguous())
    x = merge_heads(x)
    return W.linear(x, size)


def feed_forward(x, size_ff=2048, dropout=0.1, **kw):
    y = W.linear(x, size_ff, activation='relu')
    y = W.dropout(y, dropout)
    return W.linear(y, x.shape[1])


def residual_add(x, layer, dropout=0.1, **kw):
    y = W.layer_norm(x)
    y = layer(y, **kw)
    y = W.dropout(y, dropout)
    return x+y


def encoder(x, num_encoder=6, **kw):
    for i in range(num_encoder):
        x = residual_add(x, multi_head_attention, **kw)
        x = residual_add(x, feed_forward, **kw)
    return W.layer_norm(x)


def decoder(x, y, num_decoder=6, mask_x=None, mask_y=None, **kw):
    for i in range(num_decoder):
        y = residual_add(y, multi_head_attention, mask=mask_y, **kw)
        y = residual_add(x, multi_head_attention, y=y, mask=mask_x, **kw)
        y = residual_add(y, feed_forward, **kw)
    return W.layer_norm(y)


def transformer(x, y, **kw):
    x = encoder(x, **kw)
    x = decoder(x, y, **kw)
    return x


class Transformer(nn.Module):

    def __init__(self, *shape, **kw):
        super().__init__()
        self.kw = kw
        warm.engine.prepare_model_(self, *shape)
        
    def forward(self, x, y):
        return transformer(x, y, **self.kw)

```
