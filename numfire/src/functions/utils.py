from .._typing import Array as A
from ..base import MakeOP
from ...backend.backend import xp
from .primitive_reduct import sum
from ..utils import broadcast_backward
from .xpy_utils import get_dev, module
from xpy import primitive
from typing import Union

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

AType = Union[A, int, float, xp().ndarray]

def maximum(x:AType, y:AType):
    d = get_dev(x, y)

    def _fun(x, y):
        from ..array import as_nd
        _maximum = primitive(d, 'maximum')
        out = as_nd(_maximum(x, y))

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

def minimum(x:AType, y:AType):
    d = get_dev(x, y)

    def _fun(x, y):
        from ..array import as_nd
        _minimum = primitive(d, 'minimum')
        out = as_nd(_minimum(x, y))

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

def where(cond: AType, x: AType, y: AType):
    d = get_dev(x, y)

    def _fun(cond, x, y):
        from ..array import as_nd
        _wh = primitive(d, 'where')
        out = as_nd(_wh(cond, x, y))

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
