"""
Reduction operations with full autograd support.

Implements sum, mean, max, min, prod with:
    • axis=None or axis=int/tuple[int]
    • keepdims=True/False
    • NumPy or CuPy backend (xp)
    • Broadcasting-correct backward pass
"""

from __future__ import annotations
from .._typing import TensorLike
from ..base import MakeOP
from xpy import primitive
import torch

def unwrap(x):
    return getattr(x, '__backend_buffer__', x)

def maker(*args, func, nd=True):
    from ..array import as_nd
    _args = tuple(unwrap(a) for a in args)
    out = func(*_args)
    return as_nd(out) if nd else out


# ============================================================
# SUM
# ============================================================

def sum(x: TensorLike, axis=None, keepdims=False):

    def _fun(x):
        from ..array import as_nd

        out = maker(x, func=lambda x:torch.sum(x, dim=axis, keepdim=keepdims))

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

def mean(x: TensorLike, axis=None, keepdims=False, dtype=None):
    def _fun(x):
        from ..array import as_nd
        # expand_dims = module(d).expand_dims
        # broadcast_to = module(d).broadcast_to

        out = maker(x, func=lambda x:torch.mean(x, dim=axis, keepdim=keepdims))

        if axis is None:
            N = x.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(a if a >= 0 else a + x.ndim for a in axes)

            N = 1
            for a in axes:
                N *= int(x.shape[a])

        # ------------------------------------------------------

        def grad_fn(g):
            g_raw = getattr(g, "__backend_buffer__", g)

            if not keepdims and axis is not None:
                for ax in sorted(axes):
                    # g_raw = expand_dims(g_raw, ax)
                    g_raw = torch.unsqueeze(g_raw, ax)

            g_raw = g_raw / N
            g_raw = torch.broadcast_to(g_raw, x.shape)
            return as_nd(g_raw),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# ============================================================
# MAX
# ============================================================

def max(x: TensorLike, axis=None, keepdims=False):

    def _fun(x):
        # array = module(d).array
        from ..array import as_nd
        # expand_dims = module(d).expand_dims
        # broadcast_to = module(d).broadcast_to

        # x_w = as_nd(x)
        # x_raw = x_w.__backend_buffer__

        # _max = primitive(d, 'max')
        # out_raw = _max(x_raw, axis=axis, keepdims=keepdims)
        out = maker(x, func=lambda x:torch.amax(x, dim=axis, keepdim=keepdims))

        def grad_fn(g):
            g_raw = getattr(g, '__backend_buffer__', g)

            # expand out to input shape
            out_b = unwrap(out)
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = torch.unsqueeze(out_b, ax)
            out_b = torch.broadcast_to(out_b, x.shape)

            mask = (unwrap(x) == out_b)

            # IMPORTANT: denom must be computed WITHOUT autograd
            denom = torch.sum(mask, dim=axis, keepdim=True)
            denom = torch.broadcast_to(denom, x.shape)

            grad_raw = mask * (g_raw / denom)
            return as_nd(grad_raw),


        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# ============================================================
# MIN
# ============================================================

def min(x: TensorLike, axis=None, keepdims=False):

    def _fun(x):
        # array = module(d).array
        from ..array import as_nd
        # expand_dims = module(d).expand_dims
        # broadcast_to = module(d).broadcast_to

        # x_w = as_nd(x)
        # x_raw = x_w.__backend_buffer__

        # _max = primitive(d, 'max')
        # out_raw = _max(x_raw, axis=axis, keepdims=keepdims)
        out = maker(x, func=lambda x:torch.amin(x, dim=axis, keepdim=keepdims))

        def grad_fn(g):
            g_raw = getattr(g, '__backend_buffer__', g)

            # expand out to input shape
            out_b = unwrap(out)
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = torch.unsqueeze(out_b, ax)
            out_b = torch.broadcast_to(out_b, x.shape)

            mask = (unwrap(x) == out_b)

            # IMPORTANT: denom must be computed WITHOUT autograd
            denom = torch.sum(mask, dim=axis, keepdim=True)
            denom = torch.broadcast_to(denom, x.shape)

            grad_raw = mask * (g_raw / denom)
            return as_nd(grad_raw),


        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# ============================================================
# PROD
# ============================================================

def prod(x: TensorLike, axis=None, keepdims=False):

    def _fun(x):
        from ..array import as_nd
        out = maker(x, func= lambda x: torch.prod(x, dim=axis, keepdim=keepdims))

        def grad_fn(g):
            g_raw = as_nd(g).np

            out_b = unwrap(out)
            if not keepdims and axis is not None:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(axes):
                    out_b = torch.unsqueeze(out_b, ax)
            out_b = torch.broadcast_to(out_b, x.shape)

            # Safe: out / x
            eps_mask = (unwrap(x) != 0)
            grad_raw = torch.where(eps_mask, out_b / unwrap(x), 0.0)
            grad_raw = grad_raw * g_raw

            return as_nd(grad_raw),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)
