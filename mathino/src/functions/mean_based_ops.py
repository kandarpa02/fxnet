from .._typing import Array
from .primitive_reduct import mean, sum

def var(x:Array, axis=None, ddof=0, keepdims=False):
    from ..array import as_nd
    x = as_nd(x)

    _mean = mean(x, axis=axis, keepdims=keepdims)
    diff = x - _mean
    sq = diff * diff

    n = x.size if axis is None else x.shape[axis]

    return sum(sq, axis=axis, keepdims=keepdims) / (n - ddof)


def std(x, axis=None, ddof=0, keepdims=False):
    return var(x, axis=axis, ddof=ddof, keepdims=keepdims)**-2
