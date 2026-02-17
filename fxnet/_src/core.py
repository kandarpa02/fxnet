import torch
import dataclasses
import contextlib
from typing import Any, Callable
from collections.abc import Sequence
from collections import defaultdict

@dataclasses.dataclass
class Node:
    value:Any
    parents:Sequence
    vjp:Callable

    def v(self): return self.value

    def __repr__(self):
        return f"Node(v={self.v()}, p={self.parents})"
    __str__ = __repr__

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
        from .differentiate_engine import tape_dict
        args = tuple(arg if isinstance(arg, Texor) else Texor(arg)
                     for arg in args)
                     
        y, res = fwd(*args)
        tape_dict[y] = Node(y, parents=args, vjp=lambda g: bwd(g, res))
        return y
    return infunc

import functools

def primitive(f):
    func = fxwrap(f)
    vjp = None

    def _defvjp(fwd, bwd):
        nonlocal vjp
        vjp = function_vjp_wrap(fwd, bwd)

    @functools.wraps(f)
    def callback(*args):
        from .differentiate_engine import REC
        if REC and vjp is not None:
            return vjp(*args)
        return func(*args)

    callback.defvjp = _defvjp
    callback._primitive = True   
    return callback
