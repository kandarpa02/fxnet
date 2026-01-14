from .._typing import Array as A
from ..base import MakeOP
# from ..jit.placeholder import FT_Tracer
# from ..jit.utils import name
from ...backend.backend import xp
from .primitive_reduct import sum
from ..utils import broadcast_backward

def unwrap(x):
    from ..array import NDarray
    return x.np if isinstance(x, NDarray) else x

# =====================================================================
# Maximum
# =====================================================================

def maximum(x:A, y:A):
    lib = xp()

    def _fun(x, y):
        out = lib.maximum(x, y)
        from ..array import as_nd

        def grad_fn(g):
            gx = g * (x >= y)
            gy = g * (y > x)
            return (
                broadcast_backward(gx, x.shape),
                broadcast_backward(gy, y.shape),
            )
        
        return as_nd(out), (as_nd(x), as_nd(y)), grad_fn
    
    return MakeOP(_fun)(x, y)

# =====================================================================
# Minimum
# =====================================================================

def minimum(x:A, y:A):
    lib = xp()

    def _fun(x, y):
        out = lib.minimum(x, y)
        from ..array import as_nd

        def grad_fn(g):
            gx = g * (x <= y)
            gy = g * (y < x)
            return (
                broadcast_backward(gx, x.shape),
                broadcast_backward(gy, y.shape),
            )

        return as_nd(out), (as_nd(x), as_nd(y)), grad_fn
    
    return MakeOP(_fun)(x, y)

# =====================================================================
# where
# =====================================================================

def where(cond: A, x: A, y: A):
    lib = xp()

    def _fun(cond, x, y):
        out = lib.where(cond, x, y)
        from ..array import as_nd

        def grad_fn(g):
            gx = where(cond, g, as_nd(0.))
            gy = where(cond, as_nd(0.), g)

            return (
                None,  # no gradient for condition
                broadcast_backward(gx, x.shape),
                broadcast_backward(gy, y.shape),
            )

        return (
            as_nd(out),
            (as_nd(cond), as_nd(x), as_nd(y)),
            grad_fn
        )

    return MakeOP(_fun)(cond, x, y)
