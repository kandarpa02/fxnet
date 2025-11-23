from .._typing import Array as A
import numpy as np
from ..base import function
from ..utils import broadcast_backward

Array = A|int|float|np.ndarray

def add(x:Array, y:Array):
    @function
    def _fun(x, y):
        from ..array import as_nd

        def grad_fn(g):
            g1 = broadcast_backward(g, x.shape)
            g2 = broadcast_backward(g, y.shape)
            return g1, g2
        return as_nd(np.add(x, y)), (as_nd(x), as_nd(y)), grad_fn
    
    return _fun(x, y)


def subtract(x:Array, y:Array):
    @function
    def _fun(x, y):
        from ..array import as_nd

        def grad_fn(g):
            g1 = broadcast_backward(g, x.shape)
            g2 = broadcast_backward(negative(g), y.shape)
            return g1, g2
        return as_nd(np.subtract(x, y)), (as_nd(x), as_nd(y)), grad_fn
    
    return _fun(x, y)


def negative(x:Array):
    @function
    def _fun(x):
        from ..array import as_nd

        def grad_fn(g):
            g1 = broadcast_backward(negative(g), x.shape)
            return g1, 
        return as_nd(np.negative(x)), (as_nd(x),), grad_fn
    
    return _fun(x)


def multiply(x:Array, y:Array):
    @function
    def _fun(x, y):
        from ..array import as_nd
        out = as_nd(np.multiply(x, y))

        def grad_fn(g):
            g1 = broadcast_backward(multiply(g, y), x.shape)  # raw numpy
            g2 = broadcast_backward(multiply(g, x), y.shape)
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    return _fun(x, y)

def divide(x:Array, y:Array):
    @function
    def _fun(x, y):
        from ..array import as_nd
        out = as_nd(np.divide(x, y))

        def grad_fn(g):
            g1 = broadcast_backward(divide(as_nd(1.), y), x.shape)  # raw numpy
            g2 = broadcast_backward(negative(multiply(x, y**as_nd(-2))), y.shape)
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    return _fun(x, y)

def log(x:Array):
    @function
    def _fun(x):
        from ..array import as_nd
        out = as_nd(np.log(x))

        def grad_fn(g):
            return divide(1, x)
    
        return out, (as_nd(x),), grad_fn
    
    return _fun(x)

def power(x:Array, y:Array):
    @function
    def _fun(x, y):
        from ..array import as_nd
        out = as_nd(np.power(float(x), y))

        def grad_fn(g):
            g1 = broadcast_backward(y*x**(y-1), x.shape)
            g2 = broadcast_backward(multiply(out, np.log(x)), y.shape)

            return g1, g2
        return out, (as_nd(x), as_nd(y)), grad_fn
    
    return _fun(x, y)
