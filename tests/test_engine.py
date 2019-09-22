# 08-31-2019;
"""
Test cases for warm.engine.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from warm import engine


def test_set_get_default_parent():
    a = nn.Identity()
    b = nn.Identity()
    engine.set_default_parent(a)
    assert engine.get_default_parent() is a, 'get_default_parent result mismatchs set_default_parent.'
    engine.set_default_parent(b)
    assert engine.get_default_parent() is b, 'get_default_parent result mismatchs set_default_parent.'


def test_auto_name():
    a = nn.Identity()
    for i in range(10):
        assert engine._auto_name('test', a) == f'test_{i+1}', 'new calls to _auto_name failed to increment name count.'
    a(None) # test if forward pre hook is triggered to reset names
    assert engine._auto_name('test', a) == 'test_1', 'forward_pre_hook did not work.'


def test_initialize():
    a = nn.Parameter(torch.zeros(3, 4))
    b = nn.Parameter(torch.zeros(3, 4))
    c = nn.Parameter(torch.zeros(3, 4))
    torch.manual_seed(1)
    engine.initialize_(a, 'normal_')
    torch.manual_seed(1)
    nn.init.normal_(b)
    assert torch.equal(a, b), 'initialize_ with str spec did not work correctly.'
    assert not torch.equal(a, c), 'initialize_ with str spec did not work.'
    torch.manual_seed(1)
    engine.initialize_(c, nn.init.normal_)
    assert torch.equal(a, c), 'initialize_ with function spec did not work correctly.'


def test_activate():
    a = torch.randn(3, 4)
    b = copy.deepcopy(a)
    a = engine.activate(a, 'hardshrink')
    b = F.hardshrink(b)
    assert torch.equal(a, b), 'activate with str spec did not work correctly.'
    a = engine.activate(a, 'relu')
    b = F.relu(b)
    assert torch.equal(a, b), 'activate with str spec did not work correctly.'


def test_permute():
    x = torch.randn(1, 2, 3)
    y = engine.permute(x, 'BCD', 'DCB')
    assert list(y.shape) == [3, 2, 1], 'permute 3d tensor with str in_shape and str out_shape did not work correctly.'
    y = engine.permute(x, 'BCD', None)
    assert list(y.shape) == [1, 2, 3], 'permute tensor with None out_shape did not work corretly.'
    y = engine.permute(x, 'BCD', [1, 0, 2])
    assert list(y.shape) == [2, 1, 3], 'permute tensor with list out_shape did not work corretly.'
    x = torch.randn(1, 2, 3, 4)
    y = engine.permute(x, 'BCD', 'DCB')
    assert list(y.shape) == [3, 4, 2, 1], 'permute 4d tensor with str in_shape and str out_shape did not work correctly.'
    y = engine.permute(x, 'DBC', 'CDB')
    assert list(y.shape) == [4, 1, 2, 3], 'permute 4d tensor with str in_shape and str out_shape did not work correctly.'
    x = torch.randn(1, 2, 3, 4, 5)
    y = engine.permute(x, 'BDC', 'BCD')
    assert list(y.shape) == [1, 5, 2, 3, 4], 'permute 5d tensor with str in_shape and str out_shape did not work correctly.'
    x = torch.randn(1, 2)
    y = engine.permute(x, 'BDC', 'BCD')
    assert list(y.shape) == [1, 2], 'permute 2d tensor with str in_shape and str out_shape did not work correctly.'
    y = engine.permute(x, 'CBD', 'DBC')
    assert list(y.shape) == [2, 1], 'permute 2d tensor with str in_shape and str out_shape did not work correctly.'


def test_unused_kwargs():
    kw = {'unused1':0, 'unused2':0, 'base_class':0}
    unused = engine.unused_kwargs(kw)
    assert 'base_class' not in unused, 'unused_kwargs leaks used.'
    assert set(unused.keys()) == {'unused1', 'unused2'}, 'unused_kwargs did not filter kw correctly.'


def test_prepare_model_is_ready():
    class TestModel(nn.Module):
        def forward(self, x):
            x = engine.forward(x, nn.Linear, 'linear',
                base_arg=(x.shape[-1], 4, False), # in_features, out_features, bias
                in_shape=None, out_shape=None, base_shape=None,
                initialization={'weight':'ones_'}, activation=(F.dropout, {'p':1.0}), )
            return x
    x = torch.randn(1, 2, 3)
    m = TestModel()
    assert not engine.is_ready(m), 'is_ready did not work correctly.'
    engine.prepare_model_(m, x)
    assert engine.is_ready(m), 'prepare_model_ did not work correctly.'
    assert m.linear_1.bias is None, 'linear_1 should not have bias.'
    assert torch.allclose(m.linear_1.weight, torch.Tensor([1.0])), 'linear_1.weight should be initialized to all 1s.'
    y = m(x)
    assert torch.allclose(y, torch.Tensor([0.0])), 'y should be all 0s because we dropout everything.'
    assert list(y.shape) == [1, 2, 4], 'y should have shape [1, 2, 4] after linear projection.'


def test_forward():
    x = torch.randn(1, 2, 3)
    m = nn.Module()
    engine.set_default_parent(m)
    class TripleOut(nn.Module): # to test tuple_out
        def forward(self, x, b=1, c='2'):
            return x+b, x, c
    y = engine.forward(x, base_class=TripleOut, base_name='tri', tuple_out=False)
    assert isinstance(y, torch.Tensor), 'tuple_out did not work correctly.'
    y = engine.forward(x, base_class=TripleOut, base_name='tri', tuple_out=True)
    assert isinstance(y, tuple) and len(y) == 3 and y[-1] == '2', 'tuple_out did not work correctly.'
    y = engine.forward(x, base_class=TripleOut, base_name='tri', forward_kw={'c':3}, tuple_out=True)
    assert y[-1] == 3, 'forward_kw did not work correctly.'
    y = engine.forward(x, base_class=TripleOut, base_name='tri', forward_arg=(2.0,))
    assert torch.allclose(y-x, torch.Tensor([2.0])), 'forward_arg did not work correctly.'
    y = engine.forward(x, base_class=TripleOut, activation=(F.dropout, {'p':1.0}))
    assert torch.allclose(y, torch.Tensor([0.0])), 'activation did not work correctly.'
    y = engine.forward(
        x, base_class=nn.Linear, base_kw={'out_features':4}, infer_kw={'in_features':'C'}, base_shape='BDC')
    assert  y.shape[1] == 4, 'base_kw, infer_kw did not work correctly.'


def test_namespace():
    m = nn.Module()
    engine.set_default_parent(m)
    @engine.namespace
    def f1(name=''):
        return ';'.join([f2(name=name) for i in range(2)])
    @engine.namespace
    def f2(name=''):
        return name
    s0, s1, s2 = [f1() for i in range(3)]
    assert s0 == 'f1_1-f2_1;f1_1-f2_2'
    assert s1 == 'f1_2-f2_1;f1_2-f2_2'
    assert s2 == 'f1_3-f2_1;f1_3-f2_2'
