from typing import Callable, Any, Tuple, Union
from ...backend import backend as b
from ..base import TAPE_STACK, tape
from ..array import NDarray
from ...nn.parameters import Variable
from ...nn.base import Cell
from typing import Dict, Any
from ..tree_util import flatten_pytree, register_tree_node, unflatten_pytree
from ...src.functions.xpy_utils import get_dev, module
import torch

# ================================================================
# Helper functions
# ================================================================

def is_leaf(x):
    """
    Only NDarray/Variable with train=True should get gradients.
    """
    if isinstance(x, (NDarray, Variable)):
        return getattr(x, "train", False)
    return False

def expand_cell(x):
    if isinstance(x, Cell):
        return list(x.trainable_parameters())
    return None

def _extract_np(x):
    return x.__backend_buffer__ if is_leaf(x) else x

def _id(x):
    return id(x.__backend_buffer__) if is_leaf(x) else id(x) #previous, works

def _zeros_like(x):
    return torch.zeros_like(_extract_np(x))

def _ones_like(x):
    return torch.ones_like(_extract_np(x))

def norm_tuple(tpl):
    store = []
    for t in tpl:
        if not isinstance(t, tuple):
            store.append(t)
        
        else:
            store.extend(norm_tuple(t))
    return tuple(store)

# ================================================================
# BACKWARD CORE (internal)
# ================================================================
def _backward(fun, original_args, diff_leaves):
    with tape():
        out = fun(*original_args)
        if isinstance(out, (tuple, list)):
            out = out[0]

    tape_records = TAPE_STACK.pop() if TAPE_STACK else []

    grads = {
        _id(out): _ones_like(out)
    }

    for node in reversed(tape_records):

        out_id = _id(node.out)
        g = grads.get(out_id)
        if g is None:
            continue

        parent_grads = node.grad_fn(g)

        # Ensure tuple
        if not isinstance(parent_grads, tuple):
            parent_grads = (parent_grads,)

        assert len(parent_grads) == len(node.parents), (
            f"grad_fn returned {len(parent_grads)} grads "
            f"for {len(node.parents)} parents"
        )

        for parent, pg in zip(node.parents, parent_grads):
            if pg is None:
                continue

            pid = _id(parent)
            if pid in grads:
                grads[pid] = grads[pid] + pg
            else:
                grads[pid] = pg

    # ------------------------------------------------------------
    # Tape cleanup (avoid leaks)
    # ------------------------------------------------------------
    for node in tape_records:
        node.parents = None
        node.grad_fn = None
        node.out = None

    tape_records.clear()

    return out, grads



# ================================================================
# PUBLIC API: grad()
# ================================================================
def grad(fun):
    def wrapped(*args):
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
                flat_grads.append(gdict.get(gid, _zeros_like(leaf)))
            else:
                flat_grads.append(None)

        grads_tree = unflatten_pytree(flat_grads, treedef)

        return grads_tree[0] if len(args) == 1 else grads_tree

    return wrapped

backward_doc =     """
    Execute a function under tracing, build a tape of operations, and
    perform a full reverse-mode automatic differentiation pass.

    This is the heart of FakeTensor's eager-mode autograd system.

    Parameters
    ----------
    fun : Callable
        The user function whose output `out = fun(*args)` we differentiate.
        This function must use FakeTensor primitives (recorded on the tape).

    original_args : tuple
        The original unexpanded arguments passed by the user.
        These may include Cells, NDarrays, Variables, or arbitrary objects.

    diff_leaves : list
        A list of leaf nodes (NDarray/Variable) that require gradients.
        These correspond to leaves extracted after pytree flattening.

    Algorithm
    ---------
    1. Run `fun(*args)` inside a `tape()` context, collecting a linear tape.

    2. Initialize gradient dictionary:
          grads[id(out)] = ones_like(out)

       This is reverse-mode initialization (∂out/∂out = 1).

    3. Traverse all recorded Nodes in reverse order:
          for node in reversed(tape):
              g = grads[id(node.out)]
              parent_grads = node.grad_fn(g)
              accumulate into grads for each parent

    4. Return:
        (out, grads)

       where `grads` maps:
           id(leaf) → gradient array

    Notes
    -----
    • This function does NOT reshape gradients into pytrees.
      That is done by `grad()` and `value_and_grad()`.

    • This function does NOT filter which leaves to differentiate.
      That is handled before calling `_backward`.

    Returns
    -------
    tuple
        (output_of_fun, gradient_dict)
    """

grad_doc =     """
    Transform a function into one that returns gradients w.r.t. its arguments.

    This is the FakeTensor analog of:
        • JAX:  jax.grad
        • PyTorch: torch.autograd.grad but functional
        • TensorFlow: tf.GradientTape.gradient (wrapped functionally)

    Behavior
    --------
    wrapped = grad(fun)

    Calling:
        wrapped(x)
    returns the gradient of `fun(x)` with respect to x.

    If multiple arguments are passed:
        wrapped(x, y, z)
    returns a pytree of gradients matching the argument structure.

    Cell expansion
    --------------
    If an argument is a `Cell`, it is automatically replaced with its
    trainable parameters (Variables).  
    This mimics JAX pytree flattening, and allows:

        grad(loss_fn)(my_model)

    to return gradients for all parameters of `my_model`.

    Pytree semantics
    ----------------
    • Arguments are flattened using `flatten_pytree`.
    • Only leaves which satisfy `is_leaf()` receive gradients.
    • Non-leaf values produce `None`.

    Returns
    -------
    Callable
        A function returning gradients matching the structure of input args.
    """