# 08-31-2019;
"""
Test cases for warm.util.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from warm import util


def test_camel_to_snake():
    assert util.camel_to_snake('CamelAndSnake') == 'camel_and_snake'
    assert util.camel_to_snake('camelAndSnake') == 'camel_and_snake'
    assert util.camel_to_snake('camelANDSnake') == 'camel_and_snake'
    assert util.camel_to_snake('CAMELAndSnake') == 'camel_and_snake'
    assert util.camel_to_snake('CAMELAndSNAKE') == 'camel_and_snake'
    assert util.camel_to_snake('CamelAndSnake_') == 'camel_and_snake_'
    assert util.camel_to_snake('_CamelAndSnake') == '__camel_and_snake'


def test_summary_str():
    from examples.resnet import WarmResNet
    m = WarmResNet()
    s = util.summary_str(m)
    assert len(s) > 0


def test_summary():
    from examples.resnet import WarmResNet
    m = WarmResNet()
    util.summary(m)
