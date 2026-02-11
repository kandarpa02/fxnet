import torch
import numpy as np
from .basic_functions.vjps import *
from .basic_functions.vjps import _getitem

class Texor(torch.Tensor):
    __qualname__ = 'Tensor'
    __module__ = 'fxnet'

    def __hash__(self):
        return id(self)

    @staticmethod
    def __new__(cls, data, tape=None):
        data = torch.as_tensor(data).detach()
        obj = torch.Tensor._make_subclass(cls, data, require_grad=False)
        obj.tape = [] if tape is None else tape
        return obj

    def __init__(self, data, tape=None):
        pass

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # unwrap Texor -> Tensor
        def unwrap(x):
            return x.as_subclass(torch.Tensor) if isinstance(x, Texor) else x

        unwrapped_args = tuple(unwrap(a) for a in args)

        # run real torch op
        out = func(*unwrapped_args, **kwargs)

        # wrap outputs back to Texor
        def wrap(x):
            if isinstance(x, torch.Tensor):
                tx = Texor(x)
                return tx
            return x

        if isinstance(out, tuple):
            return tuple(wrap(o) for o in out)
        return wrap(out)

    
    def __add__(self, other): return add(self, other)
    def __radd__(self, other): return add(other, self)
    def __sub__(self, other): return sub(self, other)
    def __rsub__(self, other): return sub(other, self)
    def __mul__(self, other): return mul(self, other)
    def __rmul__(self, other): return mul(other, self)
    def __truediv__(self, other): return div(self, other)
    def __rtruediv__(self, other): return div(other, self)
    def __neg__(self): return neg(self)
    def __pow__(self, other): return pow(self, other)
    def __rpow__(self, other): return pow(other, self)
    def __matmul__(self, other): return matmul(self, other)
    def __rmatmul__(self, other): return matmul(other, self)
    def exp(self): return exp(self)
    def log(self): return log(self)
    def sin(self): return sin(self)
    def cos(self): return cos(self)
    def tanh(self): return tanh(self)
    def sum(self, axis=None, keepdims=False): return sum(self, axis=None, keepdims=False)
    def mean(self, axis=None, keepdims=False): return mean(self, axis=None, keepdims=False)
    def reshape(self, *shape): return reshape(self, shape)
    def view(self, *shape): return reshape(self, *shape)
    def permute(self, *axes): return permute(self, axes)
    def transpose(self, *axes): return permute(self, axes)
    def squeeze(self, axis=None): return squeeze(self, axis)
    def unsqueeze(self, axis=None): return unsqueeze(self, axis)

    def __getitem__(self, idx): return _getitem(self, idx)
    def __gt__(self, other): return greater(self, other)
    def __rgt__(self, other): return greater(other, self)
    def __ge__(self, other): return greater_equal(self, other)
    def __rge__(self, other): return greater_equal(other, self)
    def __lt__(self, other): return less(self, other)
    def __rlt__(self, other): return less(other, self)
    def __le__(self, other): return less_equal(self, other)
    def __rle__(self, other): return less_equal(other, self)
    def __eq__(self, other): return equal(self, other)
    def __req__(self, other): return equal(other, self)
    def __ne__(self, other): return not_equal(self, other)
    def __rne__(self, other): return not_equal(other, self)


