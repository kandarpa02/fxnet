from __future__ import annotations
from .._typing import Array as A
from ..base import MakeOP
from ..utils import broadcast_backward
from ...backend.backend import xp
from .primitive_array_ops import squeeze
from .primitive_reduct import max

Array = A | int | float

def equal(x: Array, y: Array):
    """
    Elementwise equality: ``x == y``.

    Returns:
        A: Boolean array.

    Autograd:
        dx = 0
        dy = 0
    """
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        out = as_nd(lib.equal(x, y))

        def grad_fn(g):
            zx = broadcast_backward(lib.zeros_like(g), x.shape)
            zy = broadcast_backward(lib.zeros_like(g), y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def not_equal(x: Array, y: Array):
    """
    Elementwise inequality: ``x != y``.
    """
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        out = as_nd(lib.not_equal(x, y))

        def grad_fn(g):
            zx = broadcast_backward(lib.zeros_like(g), x.shape)
            zy = broadcast_backward(lib.zeros_like(g), y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def less(x: Array, y: Array):
    """
    Elementwise less-than: ``x < y``.
    """
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        out = as_nd(lib.less(x, y))

        def grad_fn(g):
            zx = broadcast_backward(lib.zeros_like(g), x.shape)
            zy = broadcast_backward(lib.zeros_like(g), y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def less_equal(x: Array, y: Array):
    """
    Elementwise less-equal: ``x <= y``.
    """
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        out = as_nd(lib.less_equal(x, y))

        def grad_fn(g):
            zx = broadcast_backward(lib.zeros_like(g), x.shape)
            zy = broadcast_backward(lib.zeros_like(g), y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def greater(x: Array, y: Array):
    """
    Elementwise greater-than: ``x > y``.
    """
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        out = as_nd(lib.greater(x, y))

        def grad_fn(g):
            zx = broadcast_backward(lib.zeros_like(g), x.shape)
            zy = broadcast_backward(lib.zeros_like(g), y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def greater_equal(x: Array, y: Array):
    """
    Elementwise greater-equal: ``x >= y``.
    """
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        out = as_nd(lib.greater_equal(x, y))

        def grad_fn(g):
            zx = broadcast_backward(lib.zeros_like(g), x.shape)
            zy = broadcast_backward(lib.zeros_like(g), y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def logical_not(x: A):
    lib = xp()

    def _fun(x):
        out = lib.logical_not(x)
        from ..array import as_nd

        def grad_fn(g):
            return None  # non-differentiable

        return as_nd(out), (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)
