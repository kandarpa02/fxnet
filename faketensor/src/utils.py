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
