from collections import defaultdict
import torch
from ..tree_util import flatten_pytree, unflatten_pytree
import contextlib
from typing import ParamSpec, Union, Dict, Any

PyTree = Union[list|tuple|Dict[Any, Any]|Any]

REC = False
TAPE = None


def add_tape():
    global TAPE
    TAPE = []

def del_tape():
    global TAPE
    TAPE = None

@contextlib.contextmanager
def rec():
    global REC
    prev = REC
    REC = True
    try:
        yield
    finally:
        REC = prev

def show_tape():
    global TAPE
    return TAPE


def backward(tape):
    d = {}
    root = tape[-1]
    d[root.v()] = torch.ones_like(root.v())

    for node in reversed(tape):
        out = node.v()
        if out not in d:
            continue

        g = d[out]

        if not node.parents:
            continue

        with rec():
            parent_grads = node.vjp(g)

        for parent, pg in zip(node.parents, parent_grads):
            if parent in d:
                d[parent] = d[parent] + pg
            else:
                d[parent] = pg

    return d


class Tape:
    def __enter__(self):
        global REC
        self.rec_prev = REC
        REC = True
        add_tape()
        return self

    def __exit__(self, *args):
        global REC
        REC = self.rec_prev

    def clear(self):
        del_tape()
        
    def gradient(self, *sources:PyTree):
        flat_args, spec = flatten_pytree(sources)
        global TAPE
        tape = TAPE
        grad_dict = backward(tape)

        flat_grads = []
        for arg in flat_args:
            flat_grads.append(grad_dict.get(arg, 0.0))

        grads = unflatten_pytree(flat_grads, spec)

        return grads[0] if len(grads)==1 else grads
    
