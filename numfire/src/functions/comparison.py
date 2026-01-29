from __future__ import annotations
from .._typing import TensorLike 
from ..base import MakeOP
from ..utils import broadcast_backward
from ...backend.backend import xp
from .primitive_array_ops import squeeze
from .primitive_reduct import max
from typing import Optional
import torch

def unwrap(x):
    return getattr(x, '__backend_buffer__', x)

def maker(*args, func, nd=True):
    from ..array import as_nd
    _args = tuple(unwrap(a) for a in args)
    out = func(*_args)
    return as_nd(out) if nd else out


def equal(x: TensorLike, y: TensorLike):
    """
    Elementwise equality: ``x == y``.

    Returns:
        TensorLike: Boolean array.

    TensorLikeutograd:
        dx = 0
        dy = 0
    """

    def _fun(x, y):
        from ..array import as_nd

        out = maker(x, y, func=torch.equal)

        def grad_fn(g):
            _zeros = torch.zeros_like(unwrap(g))
            zx = broadcast_backward(_zeros, x.shape)
            zy = broadcast_backward(_zeros, y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def not_equal(x: TensorLike, y: TensorLike):
    """
    Elementwise inequality: ``x != y``.
    """
    def _fun(x, y):
        from ..array import as_nd

        out = maker(x, y, func=torch.not_equal)

        def grad_fn(g):
            _zeros = torch.zeros_like(unwrap(g))
            zx = broadcast_backward(_zeros, x.shape)
            zy = broadcast_backward(_zeros, y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def less(x: TensorLike, y: TensorLike):
    """
    Elementwise less-than: ``x < y``.
    """
    def _fun(x, y):
        from ..array import as_nd

        out = maker(x, y, func=torch.less)

        def grad_fn(g):
            _zeros = torch.zeros_like(unwrap(g))
            zx = broadcast_backward(_zeros, x.shape)
            zy = broadcast_backward(_zeros, y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def less_equal(x: TensorLike, y: TensorLike):
    """
    Elementwise less-equal: ``x <= y``.
    """
    def _fun(x, y):
        from ..array import as_nd

        out = maker(x, y, func=torch.less_equal)

        def grad_fn(g):
            _zeros = torch.zeros_like(unwrap(g))
            zx = broadcast_backward(_zeros, x.shape)
            zy = broadcast_backward(_zeros, y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def greater(x: TensorLike, y: TensorLike):
    """
    Elementwise greater-than: ``x > y``.
    """
    def _fun(x, y):
        from ..array import as_nd

        out = maker(x, y, func=torch.greater)

        def grad_fn(g):
            _zeros = torch.zeros_like(unwrap(g))
            zx = broadcast_backward(_zeros, x.shape)
            zy = broadcast_backward(_zeros, y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def greater_equal(x: TensorLike, y: TensorLike):
    """
    Elementwise greater-equal: ``x >= y``.
    """
    def _fun(x, y):
        from ..array import as_nd

        out = maker(x, y, func=torch.greater_equal)

        def grad_fn(g):
            _zeros = torch.zeros_like(unwrap(g))
            zx = broadcast_backward(_zeros, x.shape)
            zy = broadcast_backward(_zeros, y.shape)
            return zx, zy

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def logical_not(x: TensorLike):
    def _fun(x):
        from ..array import as_nd
        out = maker(x, func=torch.logical_not)

        def grad_fn(g):
            return None  # non-differentiable

        return as_nd(out), (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


def logical_and(x: TensorLike, y: TensorLike):
    def _fun(x, y):
        out = maker(x, y, func=torch.logical_and)
        from ..array import as_nd

        def grad_fn(g):
            return None, None  # non-differentiable

        return as_nd(out), (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def logical_or(x: TensorLike, y: TensorLike):
    def _fun(x, y):
        out = maker(x, y, func=torch.logical_or)
        from ..array import as_nd

        def grad_fn(g):
            return None, None

        return as_nd(out), (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


def logical_xor(x: TensorLike, y: TensorLike):
    def _fun(x, y):
        out = maker(x, y, func=torch.logical_xor)
        from ..array import as_nd

        def grad_fn(g):
            return None, None

        return as_nd(out), (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)

def logical_any(x: TensorLike, axis: Optional[int] = None, keepdims: bool = False):
    def _fun(x):
        from ..array import as_nd

        out = maker(x, func=torch.any)

        def grad_fn(g):
            return None

        return as_nd(out), (x,), grad_fn

    return MakeOP(_fun)(x)


def logical_all(x: TensorLike, axis: Optional[int] = None, keepdims: bool = False):
    def _fun(x):
        from ..array import as_nd

        out = maker(x, func=torch.all)

        def grad_fn(g):
            return None  # non-differentiable

        return as_nd(out), (x,), grad_fn

    return MakeOP(_fun)(x)
