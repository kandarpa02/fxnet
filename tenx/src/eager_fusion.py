from .base import MakeOP, TAPE_STACK
from .array import as_nd


def fuse(fun):

    def primitive(*args):

        # capture forward primitives
        local_tape = []
        TAPE_STACK.append(local_tape)
        try:
            out = fun(*args)
        finally:
            TAPE_STACK.pop()

        if not local_tape:
            return out, (), lambda g: ()

        # external parents
        produced = {id(n.out) for n in local_tape}
        parents = []
        for n in local_tape:
            for p in n.parents:
                if id(p) not in produced:
                    parents.append(p)
        parents = tuple(dict.fromkeys(parents))

        # fused backward
        def grad_fn(g):
            grads = {id(out): g}
            for n in reversed(local_tape):
                gout = grads.get(id(n.out))
                if gout is None:
                    continue
                pg = n.grad_fn(gout)
                pg = (pg,) if not isinstance(pg, tuple) else pg
                for p, gi in zip(n.parents, pg):
                    if gi is None:
                        continue
                    grads[id(p)] = grads.get(id(p), 0) + gi
            return tuple(grads.get(id(p)) for p in parents)

        return out, parents, grad_fn

    return MakeOP(primitive)
