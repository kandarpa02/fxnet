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
from xpy import primitive
import torch
Array = A
from .utils import maker, unwrap

# ============================================================
# SUM
# ============================================================

def sum(x: Array, axis=None, keepdims=False):

    def _fun(x):
        from ..array import as_nd

        out = maker(x, func=torch.sum)

        def grad_fn(g):
            g_raw = getattr(g, '__backend_buffer__', g)

            # Expand reduced dims
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    g_raw = torch.unsqueeze(g_raw, ax)

            g_raw = torch.broadcast_to(g_raw, x.shape)
            return as_nd(g_raw),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# ============================================================
# MEAN 
# ============================================================

def mean(x: Array, axis=None, keepdims=False, dtype=None):
    d = get_dev(x)

    def _fun(x):
        from ..array import as_nd
        expand_dims = module(d).expand_dims
        broadcast_to = module(d).broadcast_to

        x_w = as_nd(x)
        x_raw = x_w.__backend_buffer__
        _dtype = x_w.dtype if dtype is None else dtype

        _mean = primitive(d, 'mean')
        out_raw = _mean(x_raw, axis=axis, keepdims=keepdims)
        out = as_nd(out_raw).astype(_dtype)

        # --------- CORRECT N COMPUTATION (NO AUTOGRAD) ---------
        if axis is None:
            N = x_raw.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(a if a >= 0 else a + x_raw.ndim for a in axes)

            N = 1
            for a in axes:
                N *= int(x_raw.shape[a])

        # ------------------------------------------------------

        def grad_fn(g):
            g_raw = getattr(g, "__backend_buffer__", g)

            if not keepdims and axis is not None:
                for ax in sorted(axes):
                    g_raw = expand_dims(g_raw, ax)

            g_raw = g_raw / N
            g_raw = broadcast_to(g_raw, x_raw.shape)
            return as_nd(g_raw),

        return out, (x_w,), grad_fn

    return MakeOP(_fun)(x)


# ============================================================
# MAX
# ============================================================

def max(x: Array, axis=None, keepdims=False):
    d = get_dev(x)

    def _fun(x):
        array = module(d).array
        from ..array import as_nd
        expand_dims = module(d).expand_dims
        broadcast_to = module(d).broadcast_to

        x_w = as_nd(x)
        x_raw = x_w.__backend_buffer__

        _max = primitive(d, 'max')
        out_raw = _max(x_raw, axis=axis, keepdims=keepdims)
        out = as_nd(out_raw)

        def grad_fn(g):
            g_raw = getattr(g, '__backend_buffer__', g)

            # expand out to input shape
            out_b = out_raw
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = expand_dims(out_b, ax)
            out_b = broadcast_to(out_b, x_raw.shape)

            mask = (x_raw == out_b)

            # IMPORTANT: denom must be computed WITHOUT autograd
            denom = module(d).sum(mask, axis=axis, keepdims=True)
            denom = broadcast_to(denom, x_raw.shape)

            grad_raw = mask * (g_raw / denom)
            return as_nd(grad_raw),


        return out, (x_w,), grad_fn

    return MakeOP(_fun)(x)


# ============================================================
# MIN
# ============================================================

def min(x: Array, axis=None, keepdims=False):
    d = get_dev(x)

    def _fun(x):
        array = module(d).array
        from ..array import as_nd
        expand_dims = module(d).expand_dims
        broadcast_to = module(d).broadcast_to
        
        x_w = as_nd(x)
        x_raw = x_w.__backend_buffer__

        _min = primitive(d, 'min')
        out_raw = _min(x_raw, axis=axis, keepdims=keepdims)
        out = as_nd(out_raw)

        def grad_fn(g):
            g_raw = getattr(g, '__backend_buffer__', g)

            # Expand out_raw to input shape
            out_b = out_raw
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = expand_dims(out_b, ax)
            out_b = broadcast_to(out_b, x_raw.shape)

            # mask of minima
            mask = (x_raw == out_b)

            # IMPORTANT: use backend primitive, NOT autograd sum
            denom = module(d).sum(mask, axis=axis, keepdims=True)
            denom = broadcast_to(denom, x_raw.shape)

            grad_raw = mask * (g_raw / denom)
            return as_nd(grad_raw),

        return out, (x_w,), grad_fn

    return MakeOP(_fun)(x)


# ============================================================
# PROD
# ============================================================

def prod(x: Array, axis=None, keepdims=False):
    d = get_dev(x)

    def _fun(x):
        from ..array import as_nd
        expand_dims = module(d).expand_dims
        broadcast_to = module(d).broadcast_to
        wh = module(d).where

        x_w = as_nd(x)
        x_raw = x_w.np

        _prod = primitive(d, 'prod')
        out_raw = _prod(x_raw, axis=axis, keepdims=keepdims)
        out = as_nd(out_raw)

        def grad_fn(g):
            g_raw = as_nd(g).np

            out_b = out_raw
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = expand_dims(out_b, ax)
            out_b = broadcast_to(out_b, x_raw.shape)

            # Safe: out / x
            eps_mask = (x_raw != 0)
            grad_raw = wh(eps_mask, out_b / x_raw, 0.0)
            grad_raw = grad_raw * g_raw

            return as_nd(grad_raw),

        return out, (x_w,), grad_fn

    return MakeOP(_fun)(x)
