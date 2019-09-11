# 09-10-2019;
"""
Test cases for the warm module.
"""
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import warm


def test_warm_up():
    m = nn.Identity()
    assert not warm.engine.is_ready(m), 'is_ready did not work correctly.'
    warm.up(m, [1, 2, 3])
    assert warm.engine.is_ready(m), 'warm.up did not work correctly.'
