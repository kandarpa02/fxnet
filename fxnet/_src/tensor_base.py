import torch
import numpy as np
from .basic_functions .vjps import *

class Texor(torch.Tensor):
    __qualname__ = 'Tensor'
    __module__ = 'fxnet'

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
    def __mul__(self, other): return mul(self, other)