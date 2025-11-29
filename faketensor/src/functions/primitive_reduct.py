"""
Reduction operations with full autograd support.

Implements sum, mean, max, min, prod with:
    • axis=None or axis=int/tuple[int]
    • keepdims=True/False
    • NumPy or CuPy backend (xp)
    • Broadcasting-correct backward pass
"""

from __future__ import annotations
from .._typing import Array as A
from ..base import function
from ..utils import broadcast_backward
from ...backend.backend import xp

Array = A


# ============================================================
# SUM
# ============================================================

def sum(x: Array, axis=None, keepdims=False):
    """
    Sum over the specified axes.

    Gradient:
        Re-expand g to x.shape via broadcast.
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd
        x_nd = x
        out = as_nd(lib.sum(x_nd, axis=axis, keepdims=keepdims))

        def grad_fn(g):
            # Expand g back to original shape
            g_nd = as_nd(g)
            if not keepdims and axis is not None:
                # We must expand dims at reduced axes
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    g_nd = lib.expand_dims(g_nd, ax)
            return as_nd(lib.broadcast_to(g_nd, x_nd.shape)),

        return out, (x_nd,), grad_fn

    return function(_fun)(x)


# ============================================================
# MEAN
# ============================================================

def mean(x: Array, axis=None, keepdims=False):
    """
    Mean over the specified axes.

    Gradient:
        same as sum(x)/N  →  g / N, expanded to x.shape
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd
        x_nd = as_nd(x)

        # compute forward
        out = as_nd(lib.mean(x_nd, axis=axis, keepdims=keepdims))

        # number of elements reduced
        if axis is None:
            N = x_nd.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            dims = lib.array(x_nd.shape)[list(axes)]
            N = int(lib.prod(dims))

        def grad_fn(g):
            g_nd = as_nd(g)
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    g_nd = lib.expand_dims(g_nd, ax)
            g_nd = g_nd / N
            return as_nd(lib.broadcast_to(g_nd, x_nd.shape)),

        return out, (x_nd,), grad_fn

    return function(_fun)(x)


# ============================================================
# MAX
# ============================================================

def max(x: Array, axis=None, keepdims=False):
    """
    Maximum along axes.

    Gradient:
        Mask of elements equal to the max.
        (Subgradient: if multiple maxima, gradient is split equally.)
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd
        x_nd = as_nd(x)
        out = as_nd(lib.max(x_nd, axis=axis, keepdims=keepdims))

        def grad_fn(g):
            g_nd = as_nd(g)
            # Expand out back to x's shape
            out_b = out
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = lib.expand_dims(out_b, ax)
            out_b = lib.broadcast_to(out_b, x_nd.shape)

            mask = (x_nd == out_b)
            # Normalize when multiple max locations exist
            denom = lib.sum(mask, axis=axis, keepdims=True)
            denom = lib.broadcast_to(denom, x_nd.shape)
            grad = mask * (g_nd / denom)

            return as_nd(grad),

        return out, (x_nd,), grad_fn

    return function(_fun)(x)


# ============================================================
# MIN
# ============================================================

def min(x: Array, axis=None, keepdims=False):
    """
    Minimum along axes.

    Same gradient logic as max, but mask on minima.
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd
        x_nd = as_nd(x)
        out = as_nd(lib.min(x_nd, axis=axis, keepdims=keepdims))

        def grad_fn(g):
            g_nd = as_nd(g)
            out_b = out
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = lib.expand_dims(out_b, ax)
            out_b = lib.broadcast_to(out_b, x_nd.shape)

            mask = (x_nd == out_b)
            denom = lib.sum(mask, axis=axis, keepdims=True)
            denom = lib.broadcast_to(denom, x_nd.shape)
            grad = mask * (g_nd / denom)

            return as_nd(grad),

        return out, (x_nd,), grad_fn

    return function(_fun)(x)


# ============================================================
# PROD
# ============================================================

def prod(x: Array, axis=None, keepdims=False):
    """
    Product along axes.

    Gradient:
        d/dx_i prod(x) = prod(x) / x_i  (with zero-safe masking)
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd
        x_nd = as_nd(x)
        out = as_nd(lib.prod(x_nd, axis=axis, keepdims=keepdims))

        def grad_fn(g):
            g_nd = as_nd(g)

            # Expand out into x_nd.shape
            out_b = out
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = lib.expand_dims(out_b, ax)
            out_b = lib.broadcast_to(out_b, x_nd.shape)

            # Safe gradient: out / x, but handle x=0 carefully
            eps_mask = (x_nd != 0)
            grad = lib.where(
                eps_mask,
                out_b / x_nd,
                0.0,  # zero-safe
            )
            grad = grad * g_nd

            return as_nd(grad),

        return out, (x_nd,), grad_fn

    return function(_fun)(x)
