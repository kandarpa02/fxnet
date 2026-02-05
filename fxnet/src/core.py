import torch
import itertools
from collections.abc import Sequence
from contextlib import contextmanager
from .tensor_value import TensorBox

_trace_counter = itertools.count()


class VJPNode:
    def __init__(self, parents, vjp):
        self.parents = parents
        self.vjp = vjp


TRACE_ENABLED = True

@contextmanager
def no_trace():
    global TRACE_ENABLED
    prev = TRACE_ENABLED
    TRACE_ENABLED = False
    try:
        yield
    finally:
        TRACE_ENABLED = prev


def unwrap(x):
    if isinstance(x, TensorBox):
        return x
    return TensorBox(x)

def primitive(f, vjp_maker):
    def wrapped(*args):
        vals = [unwrap(a) for a in args]

        out = f(*vals)

        parents = [a for a in args if isinstance(a, TensorBox)]

        if not parents or not TRACE_ENABLED:
            return TensorBox(out)

        def vjp(g):
            return vjp_maker(g, *vals)

        node = VJPNode(parents, vjp)
        return TensorBox(out, node=node)

    return wrapped

def backward(t, g):
    grads = {t: g}
    stack = [t]

    while stack:
        cur = stack.pop()

        if cur.node is None:
            continue

        pgs = cur.node.vjp(grads[cur])

        for p, pg in zip(cur.node.parents, pgs):
            if p in grads:
                grads[p] += pg
            else:
                grads[p] = pg
                stack.append(p)

    return grads

from .tree_util import unflatten_pytree, flatten_pytree

from collections.abc import Sequence

def _normalize_argnums(argnums: Sequence[int] | int, nargs: int) -> tuple[int, ...]:
    if isinstance(argnums, int):
        argnums = (argnums,)
    else:
        argnums = tuple(argnums)

    norm = []
    for i in argnums:
        if i < 0:
            i = nargs + i

        if i < 0 or i >= nargs:
            raise IndexError(f"argnum {i} out of range for {nargs} arguments")

        if i in norm:
            raise ValueError(f"duplicate argnum {i}")

        norm.append(i)

    return tuple(sorted(norm))

def _grad_impl(f, args, argnums):
    trace_id = next(_trace_counter)

    boxed_args = list(args)
    arg_meta = {}
    boxed_leaves_per_arg = {}

    # ---- box selected args ----
    for i in argnums:
        leaves, meta = flatten_pytree(args[i])
        arg_meta[i] = meta

        new_leaves = [TensorBox(leaf, trace_id=trace_id) for leaf in leaves]
        boxed_leaves_per_arg[i] = new_leaves

        boxed_args[i] = unflatten_pytree(new_leaves, meta)

    # ---- forward ----
    y = f(*boxed_args)

    # ---- backward ----
    grads = backward(y, torch.ones_like(y))

    # ---- rebuild grads ----
    outs = []
    for i in argnums:
        meta = arg_meta[i]
        boxed_leaves = boxed_leaves_per_arg[i]

        g_leaves = [grads.get(t, torch.zeros_like(t)) for t in boxed_leaves]
        outs.append(unflatten_pytree(g_leaves, meta))

    g = outs[0] if len(outs) == 1 else tuple(outs)
    return y, g

def grad(f, argnum: Sequence[int] | int = 0):

    def df(*args):
        argnums = _normalize_argnums(argnum, len(args))
        _, g = _grad_impl(f, args, argnums)
        return g

    return df

def value_and_grad(f, argnum: Sequence[int] | int = 0):

    def df(*args):
        argnums = _normalize_argnums(argnum, len(args))
        y, g = _grad_impl(f, args, argnums)
        return y, g

    return df
