from .._typing import Array
from .primitive_reduct import mean, sum
from .primitive_array_ops import squeeze

def var(x: Array, axis=None, ddof=0, keepdims=False):
    from ..array import as_nd
    x = as_nd(x)

    # always keep dims for broadcasting
    _mean = mean(x, axis=axis, keepdims=True)
    diff = x - _mean
    sq = diff * diff

    if axis is None:
        n = x.size
    else:
        if isinstance(axis, tuple):
            n = 1
            for ax in axis:
                n *= x.shape[ax]
        else:
            n = x.shape[axis]

    out = sum(sq, axis=axis, keepdims=True) / float(n - ddof)

    if not keepdims:
        out = squeeze(out, axis=axis)

    return out


def std(x, axis=None, ddof=0, keepdims=False):
    return var(x, axis=axis, ddof=ddof, keepdims=keepdims)**0.5