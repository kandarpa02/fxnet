from __future__ import annotations
from .._typing import TensorLike
from ..base import MakeOP
from ...backend.backend import xp
from .utils import unwrap, maker
import torch

def unbroadcast(grad, shape):
    # Reduce extra leading dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # Reduce broadcasted dims
    axes = tuple(i for i, s in enumerate(shape) if s == 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)

    return grad


def broadcast_to(x: TensorLike, shape:tuple|list):
    """
	Broadcast a tensor to a target shape.

	Gradient:
		d/dx broadcast_to(x, shape) = unbroadcast(g, x.shape)
    """
    def _fun(x):
        from ..array import as_nd
        from .primitive_array_ops import squeeze
        out = maker(x, func=lambda x: torch.broadcast_to(x, shape))

        def grad_fn(g):
            return unbroadcast(unwrap(g), x.shape),
        
        return out, (as_nd(x),), grad_fn
    
    return MakeOP(_fun)(x)


