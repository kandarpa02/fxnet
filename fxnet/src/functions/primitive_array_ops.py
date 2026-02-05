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
from .._typing import TensorLike
from ..base import MakeOP
from ...backend.backend import xp
from .utils import unwrap, maker
import torch


# =====================================================================
# RESHAPE
# =====================================================================

def reshape(x: TensorLike, shape):
    """
    Reshape tensor to a new shape.

    Args:
        x (TensorLike): Input tensor.
        shape (tuple[int]): New shape.

    Returns:
        A: Reshaped tensor.

    Gradient:
        d/dx reshape(x, shape) = reshape(g, x.shape)
    """
    def _fun(x):
        from ..array import as_nd
        from . import reshape
        out = maker(x, func=lambda x:torch.reshape(x, shape=shape))

        def grad_fn(g):
            return reshape(g, x.shape),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# EXPAND_DIMS
# =====================================================================

def expand_dims(x: TensorLike, axis):
    """
    Insert a new axis at the specified position.

    Args:
        x (TensorLike): Input tensor.
        axis (int): Axis to insert.

    Returns:
        A: Expanded tensor.

    Gradient:
        d/dx expand_dims(x, axis) = squeeze(g, axis)
    """
    def _fun(x):
        from ..array import as_nd
        from . import squeeze
        out = maker(x, func=lambda x: torch.unsqueeze(x, axis))

        def grad_fn(g):
            return squeeze(g, axis=axis),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# SQUEEZE
# =====================================================================

def squeeze(x: TensorLike, axis=None):
    """
    Remove axes of size 1.

    Args:
        x (TensorLike): Input tensor.
        axis (int | tuple[int] | None): Axes to remove.

    Returns:
        A: Squeezed tensor.

    Gradient:
        d/dx squeeze(x, axis) = expand_dims(g, axis)
    """
    def _fun(x):
        from ..array import as_nd
        from . import expand_dims

        out = maker(x, func=lambda x: torch.squeeze(x, axis))

        def grad_fn(g):
            # Note: expand_dims requires exact axis integer or tuple.
            return expand_dims(g, axis=axis),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# CLIP
# =====================================================================

def clip(x: TensorLike, min_val, max_val):
    """
    Clip values to the range [min_val, max_val].

    Args:
        x (TensorLike): Input tensor.
        min_val (scalar or TensorLike): Lower bound.
        max_val (scalar or TensorLike): Upper bound.

    Returns:
        A: Clipped tensor.

    Gradient:
        g if x ∈ [min_val, max_val]
        0 otherwise (subgradient chosen as 0 at boundary)
    """

    def _fun(x):
        from ..array import as_nd
        from .primitive_arithmetic_and_basic_ops import multiply as mul
        out = maker(x, func=lambda x: torch.clip(x, min_val, max_val))

        def grad_fn(g):
            mask = (x >= min_val) & (x <= max_val)
            return mul(g, mask),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# ABS
# =====================================================================

def abs(x: TensorLike):
    """
    Elementwise absolute value.

    Args:
        x (TensorLike): Input tensor.

    Returns:
        A: |x|

    Gradient:
        d/dx abs(x) = sign(x)
        (subgradient at 0 chosen as 0)
    """

    def _fun(x):
        from ..array import as_nd
        from .primitive_arithmetic_and_basic_ops import sign
        out = maker(x, func=torch.abs)

        def grad_fn(g):
            return g * sign(x),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)
