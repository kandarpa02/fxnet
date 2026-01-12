"""
Shape-manipulation and common array operations with autograd support.

This module defines reshape, expand_dims, squeeze, clip, and abs for FakeTensor.
Each operation uses the `MakeOP` wrapper, storing its inputs and a
gradient function for reverse-mode autodiff.

All operations support:
    • Broadcasting (for clip/abs)
    • Higher-order gradients
    • NumPy or CuPy backend (via xp())
"""

from __future__ import annotations
from .._typing import Array as A
from ..base import MakeOP
from ..utils import broadcast_backward
from ...backend.backend import xp

Array = A   # alias

def unwrap(x):
    from ..array import NDarray
    return x.np if isinstance(x, NDarray) else x

# =====================================================================
# RESHAPE
# =====================================================================

def reshape(x: Array, shape):
    """
    Reshape tensor to a new shape.

    Args:
        x (Array): Input tensor.
        shape (tuple[int]): New shape.

    Returns:
        A: Reshaped tensor.

    Gradient:
        d/dx reshape(x, shape) = reshape(g, x.shape)
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd
        from . import reshape
        # x_nd = as_nd(x)
        out = as_nd(lib.reshape(unwrap(x), shape))

        def grad_fn(g):
            return reshape(g, x.shape),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# EXPAND_DIMS
# =====================================================================

def expand_dims(x: Array, axis):
    """
    Insert a new axis at the specified position.

    Args:
        x (Array): Input tensor.
        axis (int): Axis to insert.

    Returns:
        A: Expanded tensor.

    Gradient:
        d/dx expand_dims(x, axis) = squeeze(g, axis)
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd
        from . import squeeze
        x = unwrap(x)
        out = as_nd(lib.expand_dims(x, axis))

        def grad_fn(g):
            return squeeze(g, axis=axis),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# SQUEEZE
# =====================================================================

def squeeze(x: Array, axis=None):
    """
    Remove axes of size 1.

    Args:
        x (Array): Input tensor.
        axis (int | tuple[int] | None): Axes to remove.

    Returns:
        A: Squeezed tensor.

    Gradient:
        d/dx squeeze(x, axis) = expand_dims(g, axis)
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd
        from . import expand_dims
        x_nd = as_nd(x)
        out = as_nd(lib.squeeze(unwrap(x), axis=axis))

        def grad_fn(g):
            # Note: expand_dims requires exact axis integer or tuple.
            return expand_dims(g, axis=axis),

        return out, (x_nd,), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# CLIP
# =====================================================================

def clip(x: Array, min_val, max_val):
    """
    Clip values to the range [min_val, max_val].

    Args:
        x (Array): Input tensor.
        min_val (scalar or Array): Lower bound.
        max_val (scalar or Array): Upper bound.

    Returns:
        A: Clipped tensor.

    Gradient:
        g if x ∈ [min_val, max_val]
        0 otherwise (subgradient chosen as 0 at boundary)
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd
        from .primitive_arithmetic_and_basic_ops import multiply as mul
        x_nd = as_nd(x)
        out = as_nd(lib.clip(unwrap(x), min_val, max_val))

        def grad_fn(g):
            mask = (x_nd >= min_val) & (x_nd <= max_val)
            return mul(g, mask),

        return out, (x_nd,), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# ABS
# =====================================================================

def abs(x: Array):
    """
    Elementwise absolute value.

    Args:
        x (Array): Input tensor.

    Returns:
        A: |x|

    Gradient:
        d/dx abs(x) = sign(x)
        (subgradient at 0 chosen as 0)
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd
        x_nd = as_nd(x)
        out = as_nd(lib.abs(as_nd(x)))

        def grad_fn(g):
            return as_nd(g * lib.sign(unwrap(x))),

        return out, (x_nd,), grad_fn

    return MakeOP(_fun)(x)
