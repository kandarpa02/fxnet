from .._typing import arraytype
import numpy as np
from ..base import function
from ..utils import broadcast_backward

def add(x:arraytype, y:arraytype):
    @function
    def _fun(x, y):

        def grad_fn(g):
            g1 = broadcast_backward(g, x.shape)
            g2 = broadcast_backward(g, y.shape)
            return g1, g2
        return np.add(x, y), (x, y), grad_fn
    
    return _fun(x, y)


def mul(x, y):
    @function
    def _fun(x, y):
        out = np.multiply(x, y)

        def grad_fn(g):
            g1 = broadcast_backward(mul(g, y), x.shape)  # raw numpy
            g2 = broadcast_backward(mul(g, x), y.shape)
            return g1, g2

        return out, (x, y), grad_fn

    return _fun(x, y)
