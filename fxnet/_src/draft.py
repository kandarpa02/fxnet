import torch
import dataclasses
import contextlib
from typing import Any, Callable
from collections.abc import Sequence
from collections import defaultdict

REC = False
TAPE = []

def create_tape():
    global TAPE 
    TAPE.append([])

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
        global TAPE
        TAPE[-1].append(node)
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

def backward(tape:list[Node]):
    grads = defaultdict(lambda: 0)
    root_node_value = tape[-1].value
    grads[root_node_value] = torch.ones_like(root_node_value)

    for node in reversed(tape):
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

class EmptyTapeError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Tape:
    def __init__(self):
        pass
    
    def __enter__(self):
        global REC
        self.prev = REC
        REC = True
        create_tape()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        global REC
        REC = self.prev
        return False
    
    def clear_tape(self):
        global TAPE
        TAPE = []

    def gradient(self, *args):
        global TAPE
        if len(TAPE)==0:
            raise EmptyTapeError(
                f"Previous tape is empty. It is likely caused for using clear_tape() too early. "
                "Use clear_tape() in the last branch of the nested Tape expression only. "
                )
        gdict = backward(TAPE[-1])
        grads = tuple(gdict[a] for a in args)
        return grads[-1] if len(grads)==1 else grads


def f(x):
    y = mul(x, x)      # x^2
    z = mul(y, x)     
    return z

a = Texor(5.)

with Tape() as t2:
    with Tape() as t1:
        y = f(a)
        print(y)
        g1 = t1.gradient(a)
        print('dx', g1)
    g2 = t2.gradient(a)
    print('d2x', g2)
    t2.clear_tape()
