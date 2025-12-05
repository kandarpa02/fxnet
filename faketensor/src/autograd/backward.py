from typing import Callable, Any, Tuple, Union
from ...backend import backend as b
from ..base import TAPE_STACK, tape
from ..array import NDarray
from ...nn.parameters import Variable
from ...nn.base import Cell
from typing import Dict, Any
from ..tree_util import flatten_pytree, register_tree_node, unflatten_pytree


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
    """
    Expand a `Cell` into its underlying trainable parameters.

    Purpose
    -------
    FakeTensor supports pytree-style transformation (like JAX).
    When applying `grad()` or `value_and_grad()`, the system must know
    which components of `args` correspond to differentiable objects.

    A `Cell` is a container (like `torch.nn.Module`) that stores Variables.
    It is *not* differentiable itself, but its parameters are.

    Behavior
    --------
    • If `x` is a `Cell`, return `list(x.parameters())`.
    • Otherwise, return `None`, meaning “use x as-is”.

    This allows pytree flattening to treat:
        grad(f, cell)  →  parameters of the cell
    rather than treating Cell as undifferentiable.

    Returns
    -------
    list or None
        List of parameters if `x` is a Cell, or None otherwise.
    """
    if isinstance(x, Cell):
        return list(x.trainable_parameters())
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
    """
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
    with tape():
        out = fun(*original_args)

    tape_records = TAPE_STACK[-1] if TAPE_STACK else []

    grads = { _id(out): b.xp().ones_like(out) }

    for node in reversed(tape_records):
        g = grads.get(_id(node.out))
        if g is None:
            continue

        raw_parent_grads = node.grad_fn(g)

        # Block grads for non-trainable tensors
        parent_grads = []
        for p, pg in zip(node.parents, raw_parent_grads):
            if is_leaf(p):
                parent_grads.append(pg)
            else:
                parent_grads.append(None)

        for p, pg in zip(node.parents, parent_grads):
            if pg is None:
                continue
            pid = _id(p)
            grads[pid] = grads.get(pid, 0) + pg

    return out, grads


# ================================================================
# PUBLIC API: grad()
# ================================================================
def grad(fun):
    """
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
    """
    Create a function that returns both the value and gradient of fun(*args).

    This matches:
        • JAX:  jax.value_and_grad
        • TF:   tape.gradient + returning value
        • PyTorch: return loss, grads

    Signature
    ---------
        wrapped = value_and_grad(fun)

        y, dy = wrapped(x)

    Behavior
    --------
    • Expands Cell arguments into parameters.
    • Flattens the argument structure (pytree).
    • Runs `_backward` to compute value + gradients.
    • Reconstructs gradient pytrees matching the input args.

    argnum (currently unused)
    -------------------------
    Present for API compatibility with JAX.  
    FakeTensor currently differentiates w.r.t *all* leaves.
    Support for selective argnums can be added easily.

    Returns
    -------
    Callable
        Function returning:
            (fun(*args), gradients)
    """
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
                flat_grads.append(gdict.get(gid, b.xp().zeros_like(leaf.np)))
            else:
                flat_grads.append(None)

        grads_tree = unflatten_pytree(flat_grads, treedef)

        return out, grads_tree[0] if len(args) == 1 else grads_tree

    return wrapped
