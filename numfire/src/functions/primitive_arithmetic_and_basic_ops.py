"""
Vectorized primitive array operations with autograd support.

This module defines core numerical operations (add, multiply, matmul, etc.)
for the FakeTensor autograd system. Each operation is wrapped using `MakeOP`,
which records the forward pass, its inputs, and a gradient MakeOP that
computes partial derivatives during the backward pass.

All operations support:
    • Broadcasting (NumPy/CuPy rules)
    • Higher-order gradients (via MakeOPal closures)
    • Mixed scalar/array inputs
    • Any backend implementing the xp() interface (NumPy or CuPy)

The type `TensorLike` represents any valid input to these ops:
FakeTensor NDarray, backend arrays, or Python scalars.
"""

from __future__ import annotations
from .._typing import TensorLike
from ..base import MakeOP
from ..utils import broadcast_backward
from ...backend.backend import xp
from .primitive_array_ops import squeeze
from .primitive_reduct import max
from .utils import maximum
from ..ndarray.array_creation import zeros_like
from xpy import primitive
from .xpy_utils import get_dev, device_shift, module
from torch import tensor
import torch
from .utils import unwrap, maker

# =====================================================================
# ADD
# =====================================================================

def add(x: TensorLike, y: TensorLike):
    """
    Elementwise addition: ``x + y``.

    Args:
        x (TensorLike): First operand.
        y (TensorLike): Second operand.

    Returns:
        A: Result of elementwise addition.

    Autograd:
        dx = broadcast_backward(g, x.shape)
        dy = broadcast_backward(g, y.shape)
    """
    # d = get_dev(x, y) 

    def _fun(x, y):
        from ..array import as_nd

        out = maker(x, y, func=torch.add)

        def grad_fn(g):
            g1 = broadcast_backward(g, x.shape)
            g2 = broadcast_backward(g, y.shape)
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)

# =====================================================================
# SUBTRACT
# =====================================================================

def subtract(x: TensorLike, y: TensorLike):
    """
    Elementwise subtraction: ``x - y``.

    Args:
        x (TensorLike): Minuend.
        y (TensorLike): Subtrahend.

    Returns:
        A: Result of elementwise subtraction.
    """

    def _fun(x, y):
        from ..array import as_nd, negative
        
        out = maker(x, y, func=torch.subtract)

        def grad_fn(g):
            g1 = broadcast_backward(g, x.shape)
            g2 = broadcast_backward(negative(g), y.shape)
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


# =====================================================================
# NEGATIVE
# =====================================================================

def negative(x: TensorLike):
    """
    Elementwise negation: ``-x``.

    Args:
        x (TensorLike): Input.

    Returns:
        A: The negated tensor.
    """

    def _fun(x):
        from ..array import as_nd, negative as neg
        
        out = maker(x, func=torch.neg)

        def grad_fn(g):
            return neg(g),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# MULTIPLY
# =====================================================================

def multiply(x: TensorLike, y: TensorLike):
    """
    Elementwise multiplication: ``x * y``.

    Args:
        x (TensorLike): First operand.
        y (TensorLike): Second operand.

    Returns:
        A: Result of elementwise multiplication.
    """

    def _fun(x, y):
        from ..array import as_nd
        from . import multiply as mul  # safe recursive use
        
        out = maker(x, y, func=torch.mul)

        def grad_fn(g):
            g1 = broadcast_backward(mul(g, y), x.shape)
            g2 = broadcast_backward(mul(g, x), y.shape)
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


# =====================================================================
# DIVIDE
# =====================================================================

def divide(x: TensorLike, y: TensorLike):
    """
    Elementwise division: ``x / y``.

    Args:
        x (TensorLike): Numerator.
        y (TensorLike): Denominator.

    Returns:
        A: Result of elementwise division.
    """

    def _fun(x, y):
        from ..array import as_nd, negative
        from . import multiply as mul, power, divide as div

        out = maker(x, y, func=torch.div)

        def grad_fn(g):
            from ..ndarray.array_creation import ones_like
            g1 = broadcast_backward(mul(g, div(ones_like(y), y)), x.shape)
            g2 = broadcast_backward(
                negative(mul(g, mul(x, power(y, as_nd(-2))))),
                y.shape,
            )
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


# =====================================================================
# LOG
# =====================================================================

def log(x: TensorLike):
    """
    Natural logarithm: ``log(x)``.

    Args:
        x (TensorLike): Input tensor.

    Returns:
        A: ``log(x)``

    Autograd:
        d/dx log(x) = 1/x
    """

    def _fun(x):
        from ..array import as_nd

        eps = 1e-12
        inp = maximum(x, as_nd(eps))

        out = maker(x, func=torch.log)

        def grad_fn(g):
            return (g / x + eps),

        return out, grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# EXP
# =====================================================================

def exp(x:TensorLike):
    d = get_dev(x) 
    def _fun(x):
        from ..array import as_nd
        out = maker(x, func=torch.exp)

        def grad_fn(g):
            return (multiply(g, out),)
        
        return out, (as_nd(x), ), grad_fn
    return MakeOP(_fun)(x)


# =====================================================================
# SQRT
# =====================================================================

def sqrt(x:TensorLike):
    return x**(0.5)

# =====================================================================
# RECIPROCAL
# =====================================================================
        
def reciprocal(x:TensorLike):
    return 1/x

# =====================================================================
# Sign
# =====================================================================
def sign(x:TensorLike):
    d = get_dev(x) 
    def _fun(x):
        from ..array import as_nd
        _sign = primitive(d, 'sign')
        out = maker(x, func=torch.sign)

        def grad_fn(g):
            return (as_nd(zeros_like(x)),)
        
        return out, (as_nd(x),), grad_fn
    return MakeOP(_fun)(x)



# =====================================================================
# POWER (x ** y)
# =====================================================================

def power(x: TensorLike, y: TensorLike):
    """
    Elementwise power: ``x ** y``.

    Args:
        x (TensorLike): Base.
        y (TensorLike): Exponent.

    Returns:
        A: Result of ``x ** y``.
    """
    d = get_dev(x, y) 

    def _fun(x, y):
        from ..array import as_nd
        from . import add, subtract, multiply, log, power
        from ..ndarray.array_creation import ones_like

        out = maker(x, y, func=torch.pow)

        def grad_fn(g):
            # d/dx = y * x^(y-1)
            _one = ones_like(y)
            dx = multiply(g, multiply(y, power(x, subtract(y, _one))))
            # d/dy = (x^y) * log(x)
            dy = multiply(g, multiply(out, log(x)))

            return broadcast_backward(dx, x.shape), broadcast_backward(dy, y.shape)

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


# =====================================================================
# TRANSPOSE
# =====================================================================
from .utils import unwrap

def transpose(x: TensorLike, axes=None):
    """
    Permute tensor axes.

    Args:
        x (TensorLike): Input tensor.
        axes (tuple[int] | None): Axis permutation.

    Returns:
        A: Transposed tensor.
    """
    d = get_dev(x) 

    def _fun(x):
        from ..array import as_nd
        from . import transpose

        out = maker(x, func=torch.transpose)

        def grad_fn(g):
            if axes is None:
                rev_axes = None
            else:
                rev_axes = tuple(torch.argsort(torch.tensor(axes)))
            return transpose(g, axes=rev_axes),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# MATMUL
# =====================================================================

def matmul(a: TensorLike, b: TensorLike):
    """
    Matrix multiplication: ``a @ b``.

    Supports:
        • Vector @ Vector (dot)
        • Matrix @ Vector
        • Vector @ Matrix
        • Matrix @ Matrix
        • Batched matmul (… × M × K @ … × K × N)

    Args:
        a (TensorLike): Left operand.
        b (TensorLike): Right operand.

    Returns:
        A: The matrix product.
    """
    d = get_dev(a, b) 

    def _fun(a, b):
        from ..array import as_nd
        from . import matmul
        from .primitive_array_ops import expand_dims

        out = maker(a, b, func=torch.matmul)

        def grad_fn(g):
            A, B, G = a, b, g

            # ----------------------------
            # dA
            # ----------------------------
            if A.ndim == 1:              # vector @ matrix
                A2 = expand_dims(A, 0)  # (1, K)
                G2 = G if G.ndim > 1 else expand_dims(G, 0)
                dA = squeeze(matmul(G2, torch.swapaxes(unwrap(B), -1, -2)), 0)
            else:
                dA = G @ torch.swapaxes(unwrap(B), -1, -2)

            # ----------------------------
            # dB
            # ----------------------------
            if B.ndim == 1:              # matrix @ vector
                B2 = expand_dims(B, -1)  # (K, 1)
                G2 = G if G.ndim > 1 else expand_dims(G, -1)
                dB = squeeze(matmul(torch.swapaxes(unwrap(A), -1, -2), G2), -1)
            else:
                dB = matmul(torch.swapaxes(unwrap(A), -1, -2), G)

            return as_nd(dA), as_nd(dB)

        return out, (as_nd(a), as_nd(b)), grad_fn

    return MakeOP(_fun)(a, b)
