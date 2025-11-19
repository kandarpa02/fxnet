from typing import Callable, Any, Tuple, Union
import numpy as np
from ..base import TAPE_STACK, tape
from ..array import NDarray


def _extract_np(x):
    """Return underlying numpy array for NDarray or raw numpy."""
    if isinstance(x, NDarray):
        return x.np
    return x


def _id(x):
    """ID based on numpy buffer for NDarray."""
    if isinstance(x, NDarray):
        return id(x.np)
    return id(x)


def _zero_like(x):
    return np.zeros_like(_extract_np(x))


# ================================================================
# BACKWARD CORE (internal)
# ================================================================
def _backward(fun: Callable, args, argnums: Union[int, tuple, None]):
    """
    Perform forward + backward and return (value, grad_dict).
    grad_dict maps id(x) → numpy gradient.
    """

    # -----------------------------
    # 1) Forward pass (build tape)
    # -----------------------------
    with tape():
        output = fun(*args)

    tape_records = TAPE_STACK[-1] if TAPE_STACK else []

    # Init gradient for output
    grads = { _id(output) : np.ones_like(_extract_np(output)) }

    # -----------------------------
    # 2) Backward pass
    # -----------------------------
    for node in reversed(tape_records):
        g_out = grads.get(_id(node.out))
        if g_out is None:
            continue

        parent_grads = node.grad_fn(g_out)

        for parent, parent_grad in zip(node.parents, parent_grads):
            pid = _id(parent)

            # normalize to numpy
            pg = _extract_np(parent_grad)

            if pid in grads:
                grads[pid] = grads[pid] + pg
            else:
                grads[pid] = pg

    return output, grads


# ================================================================
# PUBLIC API: grad()
# ================================================================
def grad(fun: Callable, argnum: Union[int, tuple, None]=None) -> Callable:
    """
    JAX-like grad:
      - argnum = int → gradient wrt that argument
      - argnum = tuple → gradients wrt multiple args
      - argnum = None → gradients wrt ALL args
    """
    def wrapped(*args):
        out, gdict = _backward(fun, args, argnum)

        # -----------------------------------------------
        # Normalize argnums
        # -----------------------------------------------
        if argnum is None:
            target_ids = [ _id(a) for a in args ]
        elif isinstance(argnum, int):
            target_ids = [ _id(args[argnum]) ]
        else:
            # tuple of ints
            target_ids = [ _id(args[i]) for i in argnum ]

        # -----------------------------------------------
        # Collect gradients for requested arguments
        # -----------------------------------------------
        results = [ gdict.get(tid, _zero_like(args[i] if argnum is None else args[i]))
                    for i, tid in (
                        enumerate(target_ids) if argnum is None 
                        else zip(argnum if isinstance(argnum, tuple) else [argnum], target_ids)
                    ) ]

        if isinstance(argnum, int):
            return results[0]
        res = tuple(results) 
        return res[0] if len(res)==1 else res

    return wrapped


# ================================================================
# PUBLIC API: value_and_grad()
# ================================================================

def value_and_grad(fun: Callable, argnum: Union[int, tuple, None]=None) -> Callable:
    """
    Return (value, grads)
    """
    def wrapped(*args):
        out, gdict = _backward(fun, args, argnum)

        # Same extraction logic as grad()
        if argnum is None:
            target_ids = [ _id(a) for a in args ]
            grad_vals = tuple(gdict.get(t, _zero_like(a)) for t, a in zip(target_ids, args))
        elif isinstance(argnum, int):
            target_id = _id(args[argnum])
            grad_vals = gdict.get(target_id, _zero_like(args[argnum]))
        else:
            # tuple of indices
            target_ids = [ _id(args[i]) for i in argnum ]
            grad_vals = tuple(gdict.get(t, _zero_like(args[i])) 
                              for t, i in zip(target_ids, argnum))

        return out, grad_vals

    return wrapped
