from ..core import fxwrap
import torch

@fxwrap
def unbroadcast(shape_like, grad):
    """
    Sum grad so that it matches target_shape (reverse of broadcasting)
    """
    target_shape = shape_like.shape
    # Step 1: remove extra leading dims
    while grad.dim() > len(target_shape):
        grad = grad.sum(axis=0)

    # Step 2: sum over broadcasted axes
    for i, size in enumerate(target_shape):
        if size == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

