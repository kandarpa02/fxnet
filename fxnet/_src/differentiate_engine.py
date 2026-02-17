from collections import defaultdict
import torch
from ..tree_util import flatten_pytree, unflatten_pytree
import contextlib
from typing import ParamSpec, Union, Dict, Any
import pprint

PyTree = Union[list | tuple | Dict[Any, Any] | Any]

REC = False
tape_dict = {}

@contextlib.contextmanager
def stop_gradient():
    """
    Temporarily disable gradient recording.

    Executes the enclosed block without tracking operations on the tape.
    Useful for constants, cached computations, or preventing higher-order
    derivative creation.

    Behavior
    --------
    - Saves current recording state
    - Disables recording inside the block
    - Restores previous state on exit

    Example
    -------
    >>> with stop_gradient():
    ...     y = expensive_precompute(x)   # not differentiable
    """

    global REC
    prev = REC
    REC = False
    try:
        yield
    finally:
        REC = prev


def topo_sort(root):
    """
    Return tensors in reverse-mode dependency order.

    Performs a depth-first traversal starting from `root` following recorded
    parent links in the tape. The resulting list is a topological ordering of
    the computation graph suitable for reverse-mode differentiation.

    Parameters
    ----------
    root : Tensor
        Output tensor whose computation graph will be traversed.

    Returns
    -------
    list[Tensor]
        Nodes ordered from leaves → root (forward execution order).

    Notes
    -----
    Only tensors present in the tape are traversed. Unrecorded tensors are
    treated as leaves.
    """

    global tape_dict
    visited = set()
    order = []

    def dfs(t):
        if t in visited:
            return
        visited.add(t)

        node = tape_dict.get(t, None)
        if node:
            for p in node.parents:
                dfs(p)

        order.append(t)

    dfs(root)
    return order


def backward(root):
    """
    Execute reverse-mode automatic differentiation.

    Computes gradients of `root` with respect to all reachable tensors in the
    recorded tape using vector-Jacobian products (VJPs).

    Parameters
    ----------
    root : Tensor
        Output tensor to differentiate.

    Returns
    -------
    dict[Tensor, Tensor]
        Mapping from tensor → accumulated gradient.

    Algorithm
    ---------
    1. Seed gradient of `root` with ones_like(root)
    2. Topologically sort graph
    3. Traverse in reverse order
    4. Propagate gradients using each node's VJP rule
    5. Accumulate contributions from multiple paths

    Behavior
    --------
    - Missing nodes produce empty gradient dict
    - Gradients accumulate additively for shared parents
    - Supports higher-order differentiation if recording is enabled

    Notes
    -----
    This is a pure VJP engine: gradients flow backward as cotangents.
    """

    global tape_dict
    from .basic_functions.vjps import add
    grads = defaultdict(lambda: 0.)

    if tape_dict.get(root, None) is None:
        return grads
    
    r_node = tape_dict[root]
    grads[root] = torch.ones_like(r_node.v()) #type:ignore

    order = topo_sort(root)

    for t in reversed(order):
        node = tape_dict.get(t, None)
        if node is None:
            continue

        g = grads[t]
        parent_grads = node.vjp(g)

        for parent, pg in zip(node.parents, parent_grads):
            if parent in grads:
                grads[parent] = add(grads[parent], pg) #type:ignore
            else:
                grads[parent] = pg

    return grads

class Grad:
    """
    Reverse-mode automatic differentiation context (tape).

    `Grad` records all primitive operations executed inside its context and
    allows computing gradients of a target with respect to arbitrary source
    tensors. The context can be nested to obtain higher-order derivatives.

    Behavior
    --------
    Entering the context enables recording. Exiting restores the previous
    recording state. The global tape is cleared only when leaving the outermost
    Grad scope.

    Each call to `gradient()` performs a backward pass starting from `target`
    and returns gradients w.r.t. `sources`. Gradients are treated as ordinary
    tensors, so differentiating a gradient inside an outer `Grad` context
    produces higher-order derivatives.

    Parameters
    ----------
    target : Tensor
        Scalar or tensor value whose derivative is computed.
    sources : PyTree[Tensor]
        Tensor or nested structure (list/tuple/dict) of tensors to
        differentiate with respect to.

    Returns
    -------
    PyTree[Tensor]
        Gradients matching the structure of `sources`. If a single tensor is
        provided, the tensor gradient is returned directly.

    Notes
    -----
    - Default seed (cotangent) is 1.0 for scalar targets.
    - Missing paths in the graph return zero gradients.
    - Nested Grad contexts compute higher-order derivatives.

    Example
    -------
    >>> a = fx.constant(4.)
    >>> b = fx.constant(3.)

    >>> with fx.Grad() as g2:
    ...     with fx.Grad() as g1:
    ...         y = a ** b
    ...         da, db = g1.gradient(y, [a, b])
    ...     d2a, _ = g2.gradient(da, [a, b])
    ...     _, d2b = g2.gradient(db, [a, b])

    Computes:
        da  = ∂y/∂a
        d2a = ∂²y/∂a²
        db  = ∂y/∂b
        d2b = ∂²y/∂b²
    """

    def __enter__(self):
        global REC
        self.prev = REC
        REC = True
        return self

    def __exit__(self, *args):
        global REC, tape_dict
        REC = self.prev
        
        if not REC:
            tape_dict.clear()

        return False

    def gradient(self, target, sources: PyTree):
        """
        Compute gradients of `target` with respect to `sources`.

        Runs a reverse-mode backward pass starting from `target` and returns the
        derivatives for every tensor contained in `sources`. The structure of the
        returned gradients matches the structure of `sources`.

        Parameters
        ----------
        target : Tensor
            Output tensor to differentiate. If scalar, the implicit seed
            (cotangent) is 1.0.
        sources : PyTree[Tensor]
            A tensor or nested container (list/tuple/dict) of tensors whose
            gradients are requested.

        Returns
        -------
        PyTree[Tensor]
            Gradients matching the structure of `sources`.
            If a single tensor is given, its gradient is returned directly.

        Behavior
        --------
        - Internally flattens `sources` into a list
        - Executes `backward(target)` to build a gradient map
        - Missing entries in the graph receive zero gradients
        - Restores original container structure before returning

        Notes
        -----
        This function does not stop gradient tracking. If called inside an outer
        `Grad` context, the returned gradients remain differentiable, enabling
        higher-order derivatives.
        """

        flat_args, spec = flatten_pytree(sources)

        grad_dict = backward(target)

        flat_grads = []
        for arg in flat_args:
            flat_grads.append(grad_dict.get(arg, 0.0))

        grads = unflatten_pytree(flat_grads, spec)

        return grads[0] if len(grads)==1 else grads
    
