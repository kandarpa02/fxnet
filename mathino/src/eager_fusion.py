from contextlib import contextmanager
from .base import _RECORDING, TAPE_STACK, no_record, active_tape, Node

@contextmanager
def fuse():
    """
    Capture all primitive ops inside this block
    and collapse them into a single Node on exit.
    """
    # This is the OUTER tape (created by grad)
    outer_tape = active_tape()

    # Sub-tape for fusion
    sub_tape = []
    TAPE_STACK.append(sub_tape)

    try:
        yield
    finally:
        TAPE_STACK.pop()

    # Nothing recorded → nothing to fuse
    if not sub_tape or outer_tape is None:
        return

    # The output is the last node's output
    out = sub_tape[-1].out

    # Find external parents
    internal_outs = {n.out for n in sub_tape}
    parents = []
    for n in sub_tape:
        for p in n.parents:
            if p not in internal_outs:
                parents.append(p)
    parents = tuple(dict.fromkeys(parents))

    # Build automatic fused grad_fn
    def fused_grad_fn(g):
        grads = {out: g}

        for node in reversed(sub_tape):
            if node.out not in grads:
                continue

            with no_record():
                parent_grads = node.grad_fn(grads[node.out])

            for p, gp in zip(node.parents, parent_grads):
                grads[p] = grads.get(p, 0) + gp

        return tuple(grads.get(p, 0) for p in parents)

    outer_tape.append(Node(out, parents, fused_grad_fn))


def find_external_parents(sub_tape):
    internal_outs = {n.out for n in sub_tape}
    parents = []

    for n in sub_tape:
        for p in n.parents:
            if p not in internal_outs:
                parents.append(p)

    return tuple(dict.fromkeys(parents))  # deduplicate

def key(a):
    return a.np


def make_fused_grad_fn(sub_tape, parents, out):
    def fused_grad_fn(g):
        grads = {key(out): g}

        for node in reversed(sub_tape):
            kout = key(node.out)
            if kout not in grads:
                continue

            with no_record():
                parent_grads = node.grad_fn(grads[kout])

            for p, gp in zip(node.parents, parent_grads):
                kp = key(p)
                grads[kp] = grads.get(kp, 0) + gp

        return tuple(grads.get(key(p), 0) for p in parents)


    return fused_grad_fn

def record_fused_subgraph(out, sub_tape):
    parents = find_external_parents(sub_tape)
    grad_fn = make_fused_grad_fn(sub_tape, parents, out)

    t = active_tape()
    if t is not None:
        t.append(Node(out, parents, grad_fn))

