
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


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*BasicBlock.expansion,
                kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion), )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BasicBlock.expansion), )

    def forward(self, x):
        y = self.residual_function(x) + self.shortcut(x)
        return F.relu(y)


class ResNet(nn.Module):

    def __init__(self,
            block=BasicBlock, num_block=[2, 2, 2, 2]):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), )
        self.conv2 = self._make_stack(block, 64, num_block[0], 1)
        self.conv3 = self._make_stack(block, 128, num_block[1], 2)
        self.conv4 = self._make_stack(block, 256, num_block[2], 2)
        self.conv5 = self._make_stack(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, 1000)

    def _make_stack(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        stacks = []
        for stride in strides:
            stacks.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels*block.expansion        
        return nn.Sequential(*stacks)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.avg_pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)
        return y


resnet18 = ResNet()
```

-   The vanilla PyTorch version uses a `expansion` constant in the model class
    to track input - output size changes. It is not needed in the PyWarm version.
    As a result, the logic to determine additional projection is much more
    straightforward in PyWarm:

```Python 
# Warm
if y.shape[1] != x.shape[1]:
    # 1x1 conv to project channel size of x to y.


# Torch
if stride != 1 or in_channels != BasicBlock.expansion*out_channels:
    # 1x1 conv to project channel size of x to y.
```

----

## MobileNet

``` Python tab="Warm" linenums="1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import warm
import warm.functional as W


def conv(x, size, stride=1, expand=1, kernel=3, groups=1):
    x = W.conv(x, size, kernel, padding=(kernel-1)//2,
        stride=stride, groups=groups, bias=False, )
    return W.batch_norm(x, activation='relu6')


def bottleneck(x, size_out, stride, expand):
    size_in = x.shape[1]
    size_mid = size_in*expand
    y = conv(x, size_mid, kernel=1) if expand > 1 else x
    y = conv(y, size_mid, stride, kernel=3, groups=size_mid) # depthwise
    y = W.conv(y, size_out, kernel=1, bias=False) # pointwise linear
    y = W.batch_norm(y)
    if stride == 1 and size_in == size_out:
        y += x # residual shortcut
    return y


def conv1x1(x, *arg):
    return conv(x, *arg, kernel=1)


def pool(x, *arg):
    return x.mean([2, 3])


def classify(x, size, *arg):
    x = W.dropout(x, rate=0.2)
    return W.linear(x, size)


default_spec = (
    (None, 32, 1, 2, conv),  # t, c, n, s, operator
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
                        input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
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
