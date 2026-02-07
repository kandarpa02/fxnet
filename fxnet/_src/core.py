import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Any, Optional, Dict, Set

@dataclass
class Node:
    value: Any
    parents: List[Tuple["Node", Callable]]
    
    def __hash__(self):
        return id(self)
    

unwrap = lambda x: getattr(x, 'value', x)

def defvjp(f, vjps):
    from .tracer import Tracer
    """
    f: callable (e.g. np.add, np.multiply)
    vjps: list/tuple of VJP functions, each (g, *args) -> grad w.r.t that input
    """
    def wrapped(*args):
        vals = tuple(unwrap(arg) for arg in args)
        out = f(*vals)

        # Collect Tracer parents
        parents = []
        vjp_funcs = []

        for i, (arg, vjp_fn) in enumerate(zip(args, vjps)):
            if isinstance(arg, Tracer):
                parents.append(arg.node)
                # Store the arguments for the VJP
                def make_vjp(vjp_fn, args=args, i=i):
                    return lambda g: vjp_fn(g, *args)
                vjp_funcs.append(make_vjp(vjp_fn))

        if parents:
            node = Node(out, list(zip(parents, vjp_funcs)))
            return Tracer(out, node)
        else:
            return out

    return wrapped

