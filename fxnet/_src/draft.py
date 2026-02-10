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



@dataclasses.dataclass
class Node:
    value:Any
    parents:tuple
    vjp:Callable

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

class Primitive:
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
        
@Primitive
def add(x, y):
    return torch.add(x, y)

add.defvjp(
    lambda x, y: (torch.add(x, y), []),
    lambda g, res: (g, g)
)

@Primitive
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

class EmptyTapeError(RuntimeError):
    pass

class GradientTargetError(RuntimeError):
    pass


class GradScope:
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

    def clear(self):
        global TAPE
        TAPE = []

    def gradient(self, *targets):
        for t in targets:
            if not isinstance(t, Texor):
                raise ValueError(f"targets must be {Texor} for computing gradients. ")
            
        global TAPE

        if not TAPE:
            raise EmptyTapeError(
                "No active trace found. "
                "This usually happens if `clear()` was called too early "
                "or `gradient()` is used outside the Trace context."
            )

        tape = TAPE[-1]
        if not tape:
            raise EmptyTapeError(
                "The current trace is empty. "
                "Ensure operations were executed inside the Trace block."
            )

        gdict = backward(tape)

        grads = []
        for t in targets:
            if t not in gdict:
                raise GradientTargetError(
                    f"Cannot compute gradient for {t}. "
                    "This value was not produced inside the current Trace."
                )
            grads.append(gdict[t])

        return grads[0] if len(grads) == 1 else tuple(grads)


def f(x):
    y = mul(x, x)      # x^2
    z = mul(y, x)     
    return z

a = Texor(torch.tensor(3.))

with GradScope() as t2:
    with GradScope() as t1:
        y = f(a)
        print(y)
        g1 = t1.gradient(a)
        print('dx', g1)
    g2 = t2.gradient(a)
    print('d2x', g2)
    t2.clear()
