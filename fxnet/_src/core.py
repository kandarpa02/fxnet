import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Any, Optional, Dict, Set

@dataclass
class Node:
    value: Any
    parents: List[Tuple["Node", Callable]]
    
    def __hash__(self):
        return id(self)
    

# def unwrap(x):
#     """Extract value from Tracer or return the value itself"""
#     if hasattr(x, 'value'):
#         return x.value
#     return x

def unwrap(x):
    from .tensor_base import Tracer
    while isinstance(x, Tracer):
        x = x.value
    return x


def defvjp(f, vjps):
    from .tensor_base import Tracer
    def wrapped(*args):
        vals = [unwrap(a) for a in args]
        out = f(*vals)

        parents = []
        vjp_fns = []

        for arg, vjp in zip(args, vjps):
            if isinstance(arg, Tracer):
                parents.append(arg.node)

                def make_vjp(vjp, args=args):
                    def _vjp(g):
                        raw_args = [unwrap(a) for a in args]
                        return vjp(g, *raw_args)  # <-- RAW TENSORS ONLY
                    return _vjp

                vjp_fns.append(make_vjp(vjp))

        if parents:
            node = Node(out, list(zip(parents, vjp_fns)))
            return Tracer(out, node)
        return out

    return wrapped


def novjp(f):
    def wrapped(*args):
        nonevjps = tuple(lambda g, *args: None for _ in args)
        return defvjp(f, vjps=nonevjps)(*args)
    return wrapped