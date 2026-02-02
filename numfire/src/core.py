import itertools
_trace_counter = itertools.count()
import typing as t
from collections.abc import Sequence

class Container:
    def __init__(self, data, node=None, trace_id=None) -> None:
        self.data = data
        self.node = node
        self.trace_id = trace_id
    
    def __repr__(self):
        return str(self.data)

class VJPNode:
    def __init__(self, parents, vjp):
        self.parents = parents
        self.vjp = vjp

TRACE_ENABLED = True

from contextlib import contextmanager
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
    if isinstance(x, Container):
        return x.data
    if hasattr(x, '__backend_buffer__'):
        return x.__backend_buffer__
    return x

def primitive(f, vjp_maker):
    def wrapped(*args):
        vals = [unwrap(a) for a in args]

        out = f(*vals)

        parents = [a for a in args if isinstance(a, Container)]

        if not parents or not TRACE_ENABLED:
            return Container(out, node=None)

        def vjp(g):
            return vjp_maker(g, *vals)

        node = VJPNode(parents, vjp)
        return Container(out, node=node)

    return wrapped


def backward(t, g=1.0):
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

def grad(f, argnum: Sequence[int] | int = 0):
    # normalize argnum once
    if isinstance(argnum, int):
        argnums = (argnum,)
    else:
        argnums = tuple(argnum)

    def df(*args):
        trace_id = next(_trace_counter)

        boxed_args = []
        targets = {}  # index -> Container

        for i, val in enumerate(args):
            if i in argnums:
                t = Container(val, trace_id=trace_id)
                boxed_args.append(t)
                targets[i] = t
            else:
                boxed_args.append(val)

        y = f(*boxed_args)
        grads = backward(y)

        # collect grads in argnum order
        results = []
        for i in argnums:
            t = targets[i]
            results.append(grads.get(t, 0.0))

        return results[0] if len(results) == 1 else tuple(results)

    return df
