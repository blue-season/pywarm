
![PyWarm Logo](https://github.com/blue-season/pywarm/raw/gh-pages/docs/pywarm-logo.png)

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
(If needed, click the tabs to switch between Warm and Torch versions)


``` Python tab="Warm" linenums="1"
# powered by PyWarm
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
# vanilla PyTorch version, taken from
# pytorch tutorials/beginner_source/blitz/neural_networks_tutorial.py 
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

A couple of things you may have noticed:

-   First of all, in the PyWarm version, the entire network definition and
    data flow logic resides in the `forward()` method. You don't have to look
    up and down repeatedly to understand what `self.conv1`, `self.fc1` etc.
    is doing.

-   You do not need to track and specify `in_channels` (or `in_features`, etc.)
    for network layers. PyWarm can infer the information for you. e.g.

```Python
# Warm
x = W.conv(x, 20, 5, activation='relu')
x = W.conv(x, 50, 5, activation='relu')


# Troch
self.conv1 = nn.Conv2d(1, 20, 5, 1)
self.conv2 = nn.Conv2d(20, 50, 5, 1)
```

-   One unified `W.conv` for all 1D, 2D, and 3D cases. Fewer things to keep track of!

For deeper neural networks, see additional [examples](https://blue-season.github.io/pywarm/docs/example/).

----
## Installation

    pip3 install pywarm

----
## Quick start: 30 seconds to PyWarm

If you already have experinces with PyTorch, using PyWarm is very straightforward:

-   First, import PyWarm in you model file:
```Python
import warm
import warm.functional as W
```

-   Second, delete child module definitions in the model's `__init__()` method.
    In stead, use `W.conv`, `W.linear` ... etc. in the model's `forward()` method,
    just like how you would use torch nn functional `F.max_pool2d`, `F.relu` ... etc.

    For example, instead of writing:

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

-   You can now write in the warm way:

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

Check out the [tutorial](https://blue-season.github.io/pywarm/docs/tutorial/) 
and [examples](https://blue-season.github.io/pywarm/docs/example/) if you want to learn more!

----
## Documentation

-   [Examples](https://blue-season.github.io/pywarm/docs/example/)

-   [Tutorial](https://blue-season.github.io/pywarm/docs/tutorial/) 

-   [API reference](https://blue-season.github.io/pywarm/reference/warm/functional/)
