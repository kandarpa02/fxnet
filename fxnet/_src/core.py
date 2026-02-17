import torch
import dataclasses
import contextlib
from typing import Any, Callable
from collections.abc import Sequence
from collections import defaultdict
import functools

@dataclasses.dataclass
class Node:
    value:Any
    parents:Sequence
    vjp:Callable

    def v(self): return self.value

    def __repr__(self):
        return f"Node(v={self.v()}, p={self.parents})"
    __str__ = __repr__

def fxwrap(f):
    """
    Wrap a numerical function to automatically convert inputs to `Texor`.

    Ensures that all arguments are `Texor` objects so that they are
    compatible with the autodiff engine.

    Parameters
    ----------
    f : Callable
        Function operating on tensors or numeric values.

    Returns
    -------
    Callable
        Wrapped function where all inputs are converted to `Texor` if
        necessary before execution.
    """
    def infunc(*args):
        from .tensor_base import Texor
        args = tuple(arg if isinstance(arg, Texor) else Texor(arg)
                     for arg in args)
        return f(*args)
    return infunc


def function_vjp_wrap(fwd, bwd):
    """
    Wrap a function with its forward and backward (VJP) rules for the tape.

    Creates a wrapper that, when called, performs the forward pass to compute
    the output and any residuals, registers a `Node` in the global tape
    linking the output to its inputs, and attaches the VJP function for
    backward differentiation.

    Parameters
    ----------
    fwd : Callable
        Forward function returning (output, residuals). Residuals are any
        intermediate data needed for backward computation.
    bwd : Callable
        Backward function mapping (grad_output, residuals) → tuple of input
        gradients.

    Returns
    -------
    Callable
        Function that executes the forward pass, registers a Node, and
        supports reverse-mode differentiation via VJP.

    Notes
    -----
    - Automatically converts inputs to `Texor`.
    - The Node is stored in the global `tape_dict` keyed by output.
    - Residuals from forward pass are captured and passed to backward.
    """
    def infunc(*args):
        from .tensor_base import Texor
        from .differentiate_engine import tape_dict
        args = tuple(arg if isinstance(arg, Texor) else Texor(arg)
                     for arg in args)
                     
        y, res = fwd(*args)
        tape_dict[y] = Node(y, parents=args, vjp=lambda g: bwd(g, res))
        return y
    return infunc


def primitive(f):
    """
    Register a function as a differentiable primitive operation.

    Wraps a pure forward function and optionally attaches a custom VJP
    (vector-Jacobian product) rule. When gradient recording is active,
    calls to the function are intercepted and recorded on the tape so
    reverse-mode differentiation can propagate through it.

    Parameters
    ----------
    f : Callable
        Forward numerical function operating on tensors.

    Returns
    -------
    Callable
        Wrapped function behaving identically in eager execution but
        participating in autodiff when recording is enabled.

    Methods Added
    -------------
    defvjp(fwd, bwd)
        Attach a custom differentiation rule.

        fwd(*args) -> (output, residuals)
            Computes forward pass and returns any data required for backward.

        bwd(residuals, grad_output) -> tuple[grad_inputs]
            Computes vector-Jacobian product using saved residuals.

    Behavior
    --------
    - If gradient recording is OFF → behaves like normal function
    - If recording is ON and VJP exists → records node and uses custom rule
    - If no VJP defined → treated as non-differentiable operation

    Notes
    -----
    This decorator defines the boundary between the numerical layer and the
    autodiff engine. Higher-order derivatives are supported as long as the
    VJP implementation itself uses differentiable operations.

    Example
    -------
    >>> @primitive
    ... def square(x):
    ...     return x * x
    ...
    >>> def square_fwd(x):
    ...     y = x * x
    ...     return y, [x]
    ...
    >>> def square_bwd(g, res):
            x, = res
    ...     return (2 * x * g,)
    ...
    >>> square.defvjp(square_fwd, square_bwd)
    """

    func = fxwrap(f)
    vjp = None

    def _defvjp(fwd, bwd):
        nonlocal vjp
        vjp = function_vjp_wrap(fwd, bwd)

    @functools.wraps(f)
    def callback(*args):
        from .differentiate_engine import REC
        if REC and vjp is not None:
            return vjp(*args)
        return func(*args)

    callback.defvjp = _defvjp
    callback._primitive = True   
    return callback
