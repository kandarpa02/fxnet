import torch
import dataclasses
import contextlib
from typing import Any, Callable
from collections.abc import Sequence
from collections import defaultdict

@dataclasses.dataclass
class Node:
    value:Any
    parents:tuple
    vjp:Callable

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
        from .differentiate_engine import TAPE
        args = tuple(arg if isinstance(arg, Texor) else Texor(arg)
                     for arg in args)
                     
        y, res = fwd(*args)
        node = Node(y, parents=args, vjp=lambda g: bwd(g, res))
        TAPE.append(node)
        return y
    return infunc


class _Funcwrap:
    def __init__(self, func):
        self.func = fxwrap(func)
        self.vjp = lambda: None
    
    def defvjp(self, fwd, bwd):
        self.vjp = function_vjp_wrap(fwd, bwd)

    def __call__(self, *args):
        from .differentiate_engine import REC
        if REC:
            return self.vjp(*args)
        else:
            return self.func(*args)
        

def primitive(f):
    return _Funcwrap(f)