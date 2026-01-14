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
from ..base import MakeOP
from ...backend.backend import xp

Array = A


# ============================================================
# SUM
# ============================================================

def sum(x: Array, axis=None, keepdims=False):
    lib = xp()

    def _fun(x):
        from ..array import as_nd

        x_w = as_nd(x)
        x_raw = x_w.np

        out_raw = lib.sum(x_raw, axis=axis, keepdims=keepdims)
        out = as_nd(out_raw)

        def grad_fn(g):
            g_raw = as_nd(g).np

            # Expand reduced dims
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    g_raw = lib.expand_dims(g_raw, ax)

            g_raw = lib.broadcast_to(g_raw, x_raw.shape)
            return as_nd(g_raw),

        return out, (x_w,), grad_fn

    return MakeOP(_fun)(x)


# ============================================================
# MEAN
# ============================================================

def mean(x: Array, axis=None, keepdims=False):
    lib = xp()

    def _fun(x):
        from ..array import as_nd

        x_w = as_nd(x)
        x_raw = x_w.np

        out_raw = lib.mean(x_raw, axis=axis, keepdims=keepdims)
        out = as_nd(out_raw)

        # compute N
        if axis is None:
            N = x_raw.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            dims = [x_raw.shape[a] for a in axes]
            N = int(lib.prod(lib.array(dims)))

        def grad_fn(g):
            g_raw = as_nd(g).np

            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    g_raw = lib.expand_dims(g_raw, ax)

            g_raw = g_raw / N
            g_raw = lib.broadcast_to(g_raw, x_raw.shape)
            return as_nd(g_raw),

        return out, (x_w,), grad_fn

    return MakeOP(_fun)(x)


# ============================================================
# MAX
# ============================================================

def max(x: Array, axis=None, keepdims=False):
    lib = xp()

    def _fun(x):
        from ..array import as_nd

        x_w = as_nd(x)
        x_raw = x_w.np

        out_raw = lib.max(x_raw, axis=axis, keepdims=keepdims)
        out = as_nd(out_raw)

        def grad_fn(g):
            g_raw = as_nd(g).np

            # Expand out_raw to x_raw shape
            out_b = out_raw
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = lib.expand_dims(out_b, ax)
            out_b = lib.broadcast_to(out_b, x_raw.shape)

            mask = (x_raw == out_b)
            denom = lib.sum(mask, axis=axis, keepdims=True)
            denom = lib.broadcast_to(denom, x_raw.shape)

            grad_raw = mask * (g_raw / denom)
            return as_nd(grad_raw),

        return out, (x_w,), grad_fn

    return MakeOP(_fun)(x)


# ============================================================
# MIN
# ============================================================

def min(x: Array, axis=None, keepdims=False):
    lib = xp()

    def _fun(x):
        from ..array import as_nd

        x_w = as_nd(x)
        x_raw = x_w.np

        out_raw = lib.min(x_raw, axis=axis, keepdims=keepdims)
        out = as_nd(out_raw)

        def grad_fn(g):
            g_raw = as_nd(g).np

            out_b = out_raw
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = lib.expand_dims(out_b, ax)
            out_b = lib.broadcast_to(out_b, x_raw.shape)

            mask = (x_raw == out_b)
            denom = lib.sum(mask, axis=axis, keepdims=True)
            denom = lib.broadcast_to(denom, x_raw.shape)

            grad_raw = mask * (g_raw / denom)
            return as_nd(grad_raw),

        return out, (x_w,), grad_fn

    return MakeOP(_fun)(x)


# ============================================================
# PROD
# ============================================================

def prod(x: Array, axis=None, keepdims=False):
    lib = xp()

    def _fun(x):
        from ..array import as_nd

        x_w = as_nd(x)
        x_raw = x_w.np

        out_raw = lib.prod(x_raw, axis=axis, keepdims=keepdims)
        out = as_nd(out_raw)

        def grad_fn(g):
            g_raw = as_nd(g).np

            out_b = out_raw
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = lib.expand_dims(out_b, ax)
            out_b = lib.broadcast_to(out_b, x_raw.shape)

            # Safe: out / x
            eps_mask = (x_raw != 0)
            grad_raw = lib.where(eps_mask, out_b / x_raw, 0.0)
            grad_raw = grad_raw * g_raw

            return as_nd(grad_raw),

        return out, (x_w,), grad_fn

    return MakeOP(_fun)(x)
