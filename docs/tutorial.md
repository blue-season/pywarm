
# PyWarm Tutorial

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
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
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

    def __init__(self, block, num_block, num_classes=1000):
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
        self.fc = nn.Linear(512*block.expansion, num_classes)

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


resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
```


-   You do not need to specify a `expansion` constant in the model definition.
    In addition, the logic to determine if additional projection is needed is
    much more straightforward in PyWarm:

```Python 
# Warm
if y.shape[1] != x.shape[1]:
    # 1x1 conv to project channel size of x to y.


# Torch
if stride != 1 or in_channels != BasicBlock.expansion*out_channels:
    # 1x1 conv to project channel size of x to y.
```

