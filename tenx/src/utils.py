import numpy as np
from typing import Callable

def broadcast_backward(grad: np.ndarray, x_shape: tuple) -> np.ndarray:
    # Remove leading dims added by broadcasting
    from .functions.primitive_reduct import sum
    while len(grad.shape) > len(x_shape):
        grad = sum(grad, axis=0)

    # Reduce along broadcasted axes
    for i, (sx, sg) in enumerate(zip(x_shape, grad.shape)):
        if sx == 1 and sg != 1:
            grad = sum(grad, axis=i, keepdims=True)

    return grad

def custom_function(fun):
    from .base import function
    """
    Define custom function and its gradient rule

    A primitive must return either:

    (out, parents, grad_fn) – explicit parent list

    or:

    (out, grad_fn) – parents inferred as args

    Parameters
        fun
        Callable A Python function implementing the forward op. It should NOT mutate inputs and must return a tuple as above.

    Behavior:

        Example primitive:

        @custom_function
        def add_fun(x, y):
            out = x + y
            def grad_fn(g):
                return g, g
            return out, (x, y), grad_fn

        Calling:

            z = add_fun(x, y)

    """
    return function(fun)
