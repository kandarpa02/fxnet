from collections import defaultdict
import torch
from ..tree_util import flatten_pytree, unflatten_pytree
import contextlib
from typing import ParamSpec, Union, Dict, Any
import pprint

PyTree = Union[list|tuple|Dict[Any, Any]|Any]

REC = False

@contextlib.contextmanager
def stop_gradient():
    global REC
    prev = REC
    REC = False
    try:
        yield
    finally:
        REC = prev


def topo_sort(root):
    visited = set()
    order = []

    def dfs(t):
        if t in visited:
            return
        visited.add(t)

        node = getattr(t, "_node", None)
        if node:
            for p in node.parents:
                dfs(p)

        order.append(t)

    dfs(root)
    return order


def backward(root):
    from .basic_functions.vjps import add
    grads = defaultdict(lambda: 0.)

    if getattr(root, "_node", None) is None:
        return grads  # everything zero
    
    grads[root] = torch.ones_like(root._node.v())

    order = topo_sort(root)

    for t in reversed(order):
        node = getattr(t, "_node", None)
        if node is None:
            continue

        g = grads[t]
        parent_grads = node.vjp(g)

        # for parent, pg in zip(node.parents, parent_grads):
            # grads[parent] += pg

        for parent, pg in zip(node.parents, parent_grads):
            if parent in grads:
                grads[parent] = add(grads[parent], pg)
            else:
                grads[parent] = pg


    return grads


class Grad:
    def __enter__(self):
        global REC
        self.prev = REC
        REC = True
        return self

    def __exit__(self, *args):
        global REC
        REC = self.prev
        return False

    def gradient(self, target, sources:PyTree):
        flat_args, spec = flatten_pytree(sources)

        grad_dict = backward(target)

        flat_grads = []
        for arg in flat_args:
            flat_grads.append(grad_dict.get(arg, 0.0))

        grads = unflatten_pytree(flat_grads, spec)

        return grads[0] if len(grads)==1 else grads
    
