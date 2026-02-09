import torch
import dataclasses
import contextlib
from typing import Any, Callable
from collections.abc import Sequence
from collections import defaultdict

REC = False
@contextlib.contextmanager
def can_run_backward():
    global REC
    prev = REC
    REC = True
    try:
        yield
    finally:
        REC = prev

@dataclasses.dataclass
class Node:
    value:Any
    parents:tuple
    vjp:Callable


class Texor(torch.Tensor):
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

    
def primitive(f):
    def infunc(*args):
        args = tuple(Texor(arg) for arg in args)
        return f(*args)
    return infunc

def function_vjp_wrap(fwd, bwd):
    def infunc(*args):
        args = tuple(arg if isinstance(arg, Texor) else Texor(arg)
                     for arg in args)
        y, res = fwd(*args)
        node = Node(y, parents=args, vjp=lambda g: bwd(g, res))
        y.tape.append(node)
        return y
    return infunc

class defrules:
    def __init__(self, func):
        self.func = primitive(func)
        self.vjp = lambda: None
    
    def defvjp(self, fwd, bwd):
        self.vjp = function_vjp_wrap(fwd, bwd)

    def __call__(self, *args):
        global REC
        if REC:
            return self.vjp(*args)
        else:
            return self.func(*args)
        
@defrules
def add(x, y):
    return torch.add(x, y)

add.defvjp(
    lambda x, y: (torch.add(x, y), []),
    lambda g, res: (g, g)
)

@defrules
def mul(x, y):
    return torch.mul(x, y)

mul.defvjp(
    lambda x, y: (torch.mul(x, y), [y, x]),
    lambda g, res: (mul(g, res[0]), mul(g, res[1]))
)

a = Texor(3.)
b = Texor(5.)


def backward(root):
    grads = defaultdict(lambda: 0)
    grads[root] = torch.ones_like(root)

    for node in reversed(root.tape):
        g = grads[node.value]

        parent_grads = node.vjp(g)

        for p, gp in zip(node.parents, parent_grads):
            grads[p] = add(grads[p], gp)


    return grads

def grad(f):
    def df(*args):
        with can_run_backward():
            out = f(*args)
            grads = backward(out)

        # return grads for inputs in order
        return tuple(grads[a] for a in args)
    return df

def f(x):
    y = mul(x, x)      # x^2
    z = mul(y, x)     
    return z

g = grad(f)

x = Texor(3.)
g1, = g(x)

print(g1)     # should be 27


g2 = grad(lambda x: g(x)[0])

x = Texor(3.)
g2x, = g2(x)

print(g2x)    # should be 18
