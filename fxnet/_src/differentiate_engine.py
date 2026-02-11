from collections import defaultdict
import torch
from ..tree_util import flatten_pytree, unflatten_pytree

REC = False
TAPE = None

def create_tape():
    global TAPE 
    TAPE = []

def backward(tape:list):
    from .basic_functions.vjps import add
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
    def __init__(self, share=False) -> None:
        self.join = share

    def __enter__(self):
        global REC, TAPE
        self.prev = REC
        REC = True
        if TAPE is None or not self.join:
            create_tape()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global REC
        REC = self.prev
        if not self.join:
            global TAPE
            TAPE = None
        return False


    def gradient(self, *variables):
        flat_vars, spec = flatten_pytree(variables)

        global TAPE
        for t in flat_vars:
            from .tensor_base import Texor
            if not isinstance(t, Texor):
                raise ValueError(f"variables must be {Texor} for computing gradients. ")
            

        if not TAPE:
            raise EmptyTapeError(
                "No active trace found. "
                "This usually happens if `share` was False "
                "or `gradient()` is used outside the Trace context."
            )

        tape = TAPE
        gdict = backward(tape)

        grads = []
        for t in flat_vars:
            if t not in gdict:
                raise GradientTargetError(
                    f"Cannot compute gradient for {t}. "
                    "This value was not produced inside the current Trace."
                )
            grads.append(gdict[t])

        result = unflatten_pytree(grads, spec)

        return result[0] if len(result) == 1 else result

