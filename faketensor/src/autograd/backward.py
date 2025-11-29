from typing import Callable, Any, Tuple, Union
from ...backend import backend as b
from ..base import TAPE_STACK, tape
from ..array import NDarray
from ...neural_nets.parameters import Variable
from ...neural_nets.base import Cell
from typing import Dict, Any
from ..tree_util import flatten_pytree, register_tree_node, unflatten_pytree


# ================================================================
# Helper functions
# ================================================================

def is_leaf(x):
    """Only NDarray/Variable are differentiable leaves. Cell is NOT."""
    return getattr(x, "__is_leaf__", False)


def expand_cell(x):
    """
    Expand Cell into its parameters.
    If x is Cell -> replace it in pytree with x.parameters()
    Otherwise return None.
    """
    if isinstance(x, Cell):
        return list(x.parameters())
    return None


def _extract_np(x):
    return x.np if is_leaf(x) else x


def _id(x):
    return id(x.np) if is_leaf(x) else id(x)


def _zero_like(x):
    return b.xp().zeros_like(_extract_np(x))


# ================================================================
# BACKWARD CORE (internal)
# ================================================================
def _backward(fun, original_args, diff_leaves):

    with tape():
        out = fun(*original_args)

    tape_records = TAPE_STACK[-1] if TAPE_STACK else []

    grads = { _id(out): b.xp().ones_like(out) }

    for node in reversed(tape_records):
        g = grads.get(_id(node.out))
        if g is None:
            continue

        parent_grads = node.grad_fn(g)

        for p, pg in zip(node.parents, parent_grads):
            pid = _id(p)
            grads[pid] = grads.get(pid, 0) + pg

    return out, grads


# ================================================================
# PUBLIC API: grad()
# ================================================================
def grad(fun):
    def wrapped(*args):

        # -------------------------------------------------------
        # Expand Cells into their parameters before flattening
        # -------------------------------------------------------
        expanded_args = []
        for a in args:
            repl = expand_cell(a)
            expanded_args.append(repl if repl is not None else a)

        leaves, treedef = flatten_pytree(expanded_args)
        diff_leaves = [x for x in leaves if is_leaf(x)]

        out, gdict = _backward(fun, args, diff_leaves)

        flat_grads = []
        for leaf in leaves:
            if is_leaf(leaf):
                gid = _id(leaf)
                flat_grads.append(gdict.get(gid, b.xp().zeros_like(leaf.np)))
            else:
                flat_grads.append(None)

        grads_tree = unflatten_pytree(flat_grads, treedef)

        return grads_tree[0] if len(args) == 1 else grads_tree

    return wrapped


# ================================================================
# PUBLIC API: value_and_grad()
# ================================================================
def value_and_grad(fun: Callable, argnum: Union[int, tuple, list, None] = None) -> Callable:

    def wrapped(*args):

        # -------------------------------------------------------
        # Expand Cells into list of parameters
        # -------------------------------------------------------
        expanded_args = []
        for a in args:
            repl = expand_cell(a)
            expanded_args.append(repl if repl is not None else a)

        # Normal flatten
        leaves, treedef = flatten_pytree(expanded_args)
        diff_leaves = [x for x in leaves if is_leaf(x)]

        out, gdict = _backward(fun, args, diff_leaves)

        flat_grads = []
        for leaf in leaves:
            if is_leaf(leaf):
                gid = _id(leaf)
                flat_grads.append(gdict.get(gid, b.xp().zeros_like(leaf.np)))
            else:
                flat_grads.append(None)

        grads_tree = unflatten_pytree(flat_grads, treedef)

        return out, grads_tree[0] if len(args) == 1 else grads_tree

    return wrapped
