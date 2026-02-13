import torch
import dataclasses
import contextlib
from typing import Any, Callable
from collections.abc import Sequence
from collections import defaultdict


def fxwrap(f):
    def infunc(*args):
        from .tensor_base import Texor
        args = tuple(arg if isinstance(arg, Texor) else Texor(arg)
                     for arg in args)
        return f(*args)
    return infunc


def function_vjp_wrap(fwd, bwd):
    def infunc(*args):
        from .tensor_base import Texor
        from .tracer import Tracer, new_ids, Node
        args = tuple(arg if isinstance(arg, Texor) else Texor(arg)
                     for arg in args)
                     
        y, res = fwd(*args)
        ids = tuple(new_ids() for _ in args)
        tracers = tuple(Tracer(id) for id in ids)
        
        teaced = Tracer(id)
        

        return y
    return infunc


class _Funcwrap:
    def __init__(self, func):
        self.func = fxwrap(func)
        self.vjp = None
    
    def defvjp(self, fwd, bwd):
        self.vjp = function_vjp_wrap(fwd, bwd)

    def __call__(self, *args):
        from .differentiate_engine import REC
        if REC:
            if not self.vjp is None:
                return self.vjp(*args)
            return self.func(*args)
        else:
            return self.func(*args)
        

def primitive(f):
    return _Funcwrap(f)