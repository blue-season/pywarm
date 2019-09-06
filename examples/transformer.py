# 09-05-2019;
"""
Transformer from the paper *Attention is all you need*.
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


def multi_head_attention(x, y=None, size_v=None, num_head=1, dropout=0., bias=True):
    y = y or x
    size = x.shape[1]
    size_v = size_v or size
    q = W.linear(x)
    k = W.linear(y)
    v = W.linear(y)


def feed_forward():
    pass


def encoder(x, size_model=512, num_head=8, size_ffn=2048, dropout=0.1, activation='relu'):
    pass


def decoder():
    pass


def transformer():
    pass


