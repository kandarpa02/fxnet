from .._typing import TensorLike
from ..base import MakeOP
from ...backend.backend import xp
from .primitive_reduct import sum
from ..utils import broadcast_backward
from .xpy_utils import get_dev, module
from xpy import primitive
from typing import Union
import torch
import numpy as np

def unwrap(x):
    return getattr(x, '__backend_buffer__', x)

def maker(*args, func, nd=True):
    from ..array import as_nd
    _args = tuple(unwrap(a) for a in args)
    out = func(*_args)
    return as_nd(out) if nd else out

# =====================================================================
# Maximum
# =====================================================================


def maximum(x:TensorLike, y:TensorLike):
    def _fun(x, y):
        from ..array import as_nd
        out = maker(x, y, func=torch.maximum)

        def grad_fn(g):
            gx = g * (x >= y)
            gy = g * (y > x)
            return (
                broadcast_backward(gx, x.shape),
                broadcast_backward(gy, y.shape),
            )
        
        return out, (as_nd(x), as_nd(y)), grad_fn
    
    return MakeOP(_fun)(x, y)

# =====================================================================
# Minimum
# =====================================================================

def minimum(x:TensorLike, y:TensorLike):
    def _fun(x, y):
        from ..array import as_nd
        out = maker(x, y, func=torch.minimum)

        def grad_fn(g):
            gx = g * (x >= y)
            gy = g * (y > x)
            return (
                broadcast_backward(gx, x.shape),
                broadcast_backward(gy, y.shape),
            )
        
        return out, (as_nd(x), as_nd(y)), grad_fn
    
    return MakeOP(_fun)(x, y)

# =====================================================================
# where
# =====================================================================

def where(cond: TensorLike, x: TensorLike, y: TensorLike):

    def _fun(cond, x, y):
        from ..array import as_nd
        out = maker(cond, x, y, func=torch.where)

        def grad_fn(g):
            gx = where(cond, g, as_nd(0.))
            gy = where(cond, as_nd(0.), g)

            return (
                None,  # no gradient for condition
                broadcast_backward(gx, x.shape),
                broadcast_backward(gy, y.shape),
            )

        return (
            out,
            (as_nd(cond), as_nd(x), as_nd(y)),
            grad_fn
        )

    return MakeOP(_fun)(cond, x, y)
