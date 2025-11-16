import numpy as np
from typing import NamedTuple
from ..src.autograd.tape_recorder import Node

def opr(*args, fn):
    data = fn(*[arg.d for arg in args])
    return NDarray(data)

class NDarray:
    def __init__(self, data):
        self.d = np.asarray(data)
        self.tensornode = None
        self.grad = np.array(0.0)

    def __repr__(self):
        return self.d.__repr__()

    def __str__(self):
        return self.d.__str__()

    def __add__(self, other):
        out = opr(self, other, fn=lambda a, b: a+b)
        out.tensornode = Node(out, parents=(self, other), backward=lambda grad: (grad, grad))
        return out
