
![PyWarm Logo](docs/pywarm-logo.png)

# PyWarm

A cleaner way to build neural networks for PyTorch.

## Introduction

PyWarm is a high-level neural network construction API for PyTorch.
It only aims to simplify the network definition, and does not cover
model training, validation and data handling.

With PyWarm, you can put *all* network data flow logic in the `forward()` method of
your model, without the need to define children modules in the `__init__()` method.
This result in a much readable model definition in fewer lines of code.

----

For example, a convnet for MNIST:
(Click the tabs to switch between Warm and Torch versions)


``` Python tab="Warm" linenums="1"
import torch.nn as nn
import torch.nn.functional as F
import warm
import warm.functional as W


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        warm.engine.prepare_model_(self, [1, 1, 28, 28])

    def forward(self, x):
        x = W.conv(x, 20, 5, activation='relu')
        x = F.max_pool2d(x, 2)
        x = W.conv(x, 50, 5, activation='relu')
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 800)
        x = W.linear(x, 500, activation='relu')
        x = W.linear(x, 10)
        return F.log_softmax(x, dim=1)
```

``` Python tab="Torch" linenums="1"
# from pytorch tutorials/beginner_source/blitz/neural_networks_tutorial.py 
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

----

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

----

A couple of things you may have noticed:

-   First of all, in the PyWarm version, the entire network definition and
    data flow logic resides in the `forward()` method. You don't have to look
    up and down repeatedly to understand what `self.conv1`, `self.fc` etc.
    is doing.

-   You do not need to track and specify `in_channels` (or `in_features`, etc.)
    for network layers. PyWarm can infer the information for you. e.g.

```Python
# Warm
y = W.conv(x, 64, 7, stride=2, padding=3, bias=False)
y = W.batch_norm(y, activation='relu')


# Troch
self.conv1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True), )
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

-   One unified `W.conv` and `W.batch_norm` for all 1D, 2D, and 3D cases.
    Fewer things to keep track of!


----
## Quick start: 30 seconds to PyWarm

If you already have experinces with PyTorch, using PyWarm is straightforward:

-   First, import PyWarm in you model file:
```Python
import warm
import warm.functional as W
```

-   Second, delete child module definitions in the model's `__init__()` method.
    In stead, use `W.conv`, `W.linear` ... etc. in the model's `forward()` method,
    Pretty much like how you would use `F.max_pool2d`, `F.relu` ... etc.

    For example, instead of the old way:

```Python
# Torch
class MyModule(nn.Module):
    def __init__(self):
        ...
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        ...
    def forward(self, x):
        x = self.conv1(x)
```

-   You now write in the warm way:

```Python
# Warm
class MyWarmModule(nn.Module):
    def __init__(self):
        ...
        warm.engine.prepare_model_(self, input_shape_or_data)
    def forward(self, x):
        x = W.conv(x, out_channels, kernel_size) # no in_channels needed
```

-   Finally, don't forget to warmify the model by adding
    
    `warm.engine.prepare_model_(self, input_shape_or_data)`

    at the end of the model's `__init__()` method. You need to supply
    `input_shape_or_data`, which is either a tensor of input data, 
    or just its shape, e.g. `[1, 1, 28, 28]` for MNIST inputs.
    
    The model is now ready to use, just like any other PyTorch models.


----
## Installation

    pip3 install pywarm

