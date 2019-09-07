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


def multi_head_attention(x, y=None, num_head=8, dropout=0.1, mask=0.0, **kw):
    def split_heads(t): # (B, C, L) -> (B, N, H, L) where N*H == C
        return t.reshape(batch, num_head, size//num_head, t.shape[-1])
    def combine_heads(t): # (B, N, H, L) -> (B, C, L)
        return t.reshape(batch, -1, t.shape[-1]) # (B, C, L)
    if y is None:
        y = x
    batch, size = x.shape[:2]
    assert size%num_head == 0, 'num_head must be a divisor of size.'
    assert y.shape[:2] == x.shape[:2], 'x, y must have matching first 2 dims.'
    q = W.linear(x, size) # query
    k = W.linear(y, size) # key
    v = W.linear(y, size) # value
    q = split_heads(q) # (B, N, H, Lx)
    k = split_heads(k) # (B, N, H, Ly)
    v = split_heads(v) # (B, N, H, Ly)
    q *= (size//num_head)**(-0.5)
    a = q.transpose(2, 3).contiguous().matmul(k) # attention weights, (B, N, Lx, Ly)
    a += mask
    a = F.softmax(a, dim=-1)
    a = W.dropout(a, dropout)
    x = a.transpose(2, 3).contiguous().matmul(v) # (B, N, H, Lx)
    x = combine_heads(x) # (B, C, Lx)
    return W.linear(x)


def feed_forward(x, size_ff=2048, dropout=0.1, **kw):
    y = W.Linear(x, size_ff, activation='relu')
    y = W.dropout(y, dropout)
    return W.Linear(y, x.shape[1])


def add_shortcut(x, layer, **kw):
    y = W.layer_norm(x)
    y = layer(y, **kw)
    y = W.dropout(y)
    return x+y


def encoder(x, num_stack=6, **kw):
    for i in range(num_stack):
        x = add_shortcut(x, multi_head_attention)
        x = add_shortcut(x, feed_forward)
    return W.layer_norm(x)


def decoder(x, y, num_stack=6, mask_x=0.0, mask_y=0.0, **kw):
    for i in range(num_stack):
        x = add_shortcut(x, multi_head_attention, mask=mask_x)
        x = add_shortcut(x, multi_head_attention, y=y, mask=mask_y)
        x = add_shortcut(x, feed_forward)
    return W.layer_norm(x)


def transformer(x, y, **kw):
    x = encoder(x, **kw)
    x = decoder(x, y, **kw)
    return x


def positional_encoding():
    pass


def causal_additive_mask():
    pass


