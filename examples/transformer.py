# 09-05-2019;
"""
The Transformer model from paper *Attention is all you need*.
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


def multi_head_attention(x, y=None, num_head=8, dropout=0.1, mask=None, **kw):
    def split_heads(t): # (B, C, L) -> (B, N, H, L) where N*H == C
        return t.reshape(batch, num_head, size//num_head, t.shape[-1])
    def merge_heads(t): # (B, N, H, L) -> (B, C, L)
        return t.reshape(batch, -1, t.shape[-1]) # (B, C, L)
    if y is None:
        y = x # self attention
    batch, size = x.shape[:2] # B, C, Lx
    assert size%num_head == 0, 'num_head must be a divisor of size.'
    assert y.shape[:2] == x.shape[:2], 'The first 2 dims of x, y must match.'
    q = W.linear(x, size) # query
    k = W.linear(y, size) # key
    v = W.linear(y, size) # value
    q = split_heads(q) # (B, N, H, Lx)
    k = split_heads(k) # (B, N, H, Ly)
    v = split_heads(v) # (B, N, H, Ly)
    q *= (size//num_head)**(-0.5)
    a = q.transpose(2, 3).contiguous().matmul(k) # attention weights, (B, N, Lx, Ly)
    if mask is not None:
        a += mask
    a = F.softmax(a, dim=-1)
    a = W.dropout(a, dropout)
    x = v.matmul(a.transpose(2, 3).contiguous()) # (B, N, H, Lx)
    x = merge_heads(x) # (B, C, Lx)
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
        warm.up(self, *shape)
    def forward(self, x, y):
        return transformer(x, y, **self.kw)
