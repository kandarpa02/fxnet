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

The type `Array` represents any valid input to these ops:
FakeTensor NDarray, backend arrays, or Python scalars.
"""

from __future__ import annotations
from .._typing import Array as A
from ..base import MakeOP
from ..utils import broadcast_backward
from ...backend.backend import xp
from .primitive_array_ops import squeeze
from .primitive_reduct import max

# Allow scalars as valid inputs
Array = A | int | float
from .utils import unwrap

# =====================================================================
# ADD
# =====================================================================

def add(x: Array, y: Array):
    """
    Elementwise addition: ``x + y``.

    Args:
        x (Array): First operand.
        y (Array): Second operand.

    Returns:
        A: Result of elementwise addition.

    Autograd:
        dx = broadcast_backward(g, x.shape)
        dy = broadcast_backward(g, y.shape)
    """
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd

        out = as_nd(lib.add(x, y))

        def grad_fn(g):
            g1 = broadcast_backward(g, x.shape)
            g2 = broadcast_backward(g, y.shape)
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


# =====================================================================
# SUBTRACT
# =====================================================================

def subtract(x: Array, y: Array):
    """
    Elementwise subtraction: ``x - y``.

    Args:
        x (Array): Minuend.
        y (Array): Subtrahend.

    Returns:
        A: Result of elementwise subtraction.
    """
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd, negative

        out = as_nd(lib.subtract(x, y))

        def grad_fn(g):
            g1 = broadcast_backward(g, x.shape)
            g2 = broadcast_backward(negative(g), y.shape)
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


# =====================================================================
# NEGATIVE
# =====================================================================

def negative(x: Array):
    """
    Elementwise negation: ``-x``.

    Args:
        x (Array): Input.

    Returns:
        A: The negated tensor.
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd, negative as neg

        out = as_nd(lib.negative(x))

        def grad_fn(g):
            return neg(g),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# MULTIPLY
# =====================================================================

def multiply(x: Array, y: Array):
    """
    Elementwise multiplication: ``x * y``.

    Args:
        x (Array): First operand.
        y (Array): Second operand.

    Returns:
        A: Result of elementwise multiplication.
    """
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd
        from . import multiply as mul  # safe recursive use

        out = as_nd(lib.multiply(x, y))

        def grad_fn(g):
            g1 = broadcast_backward(mul(g, y), x.shape)
            g2 = broadcast_backward(mul(g, x), y.shape)
            return g1, g2

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


# =====================================================================
# DIVIDE
# =====================================================================

def divide(x: Array, y: Array):
    """
    Elementwise division: ``x / y``.

    Args:
        x (Array): Numerator.
        y (Array): Denominator.

    Returns:
        A: Result of elementwise division.
    """
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd, negative
        from . import multiply as mul, power, divide as div

        out = as_nd(lib.divide(x, y))

        def grad_fn(g):
            g1 = broadcast_backward(mul(g, div(as_nd(1.0), y)), x.shape)
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

def log(x: Array):
    """
    Natural logarithm: ``log(x)``.

    Args:
        x (Array): Input tensor.

    Returns:
        A: ``log(x)``

    Autograd:
        d/dx log(x) = 1/x
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd

        eps = 1e-12
        inp = lib.maximum(x, eps)
        out = as_nd(lib.log(inp))  # clamp

        def grad_fn(g):
            return (g / (x + eps),)

        return out, (as_nd(x),), grad_fn


    return MakeOP(_fun)(x)

# =====================================================================
# EXP
# =====================================================================

def exp(x:Array):
    lib = xp()
    def _fun(x):
        from ..array import as_nd
        out = as_nd(lib.exp(x))
        def grad_fn(g):
            return (multiply(g, out),)
        
        return out, (as_nd(x), ), grad_fn
    return MakeOP(_fun)(x)


# =====================================================================
# SQRT
# =====================================================================

def sqrt(x:Array):
    return x**(0.5)

# =====================================================================
# RECIPROCAL
# =====================================================================
        
def reciprocal(x:Array):
    return 1/x

# =====================================================================
# Sign
# =====================================================================
def sign(x:Array):
    lib = xp()
    def _fun(x):
        from ..array import as_nd
        out = as_nd(lib.sign(x))
        def grad_fn(g):
            return (as_nd(lib.zeros_like(x)),)
        
        return out, (as_nd(x),), grad_fn
    return MakeOP(_fun)(x)



# =====================================================================
# POWER (x ** y)
# =====================================================================

def power(x: Array, y: Array):
    """
    Elementwise power: ``x ** y``.

    Args:
        x (Array): Base.
        y (Array): Exponent.

    Returns:
        A: Result of ``x ** y``.
    """
    lib = xp()

    def _fun(x, y):
        from ..array import as_nd
        from . import add, subtract, multiply, log, power

        # if y<0:
        #     x.astype('float32')

        out = as_nd(lib.power(x, y))

        def grad_fn(g):
            # d/dx = y * x^(y-1)
            dx = multiply(g, multiply(y, power(x, subtract(y, as_nd(1)))))
            # d/dy = (x^y) * log(x)
            dy = multiply(g, multiply(out, log(x)))

            return broadcast_backward(dx, x.shape), broadcast_backward(dy, y.shape)

        return out, (as_nd(x), as_nd(y)), grad_fn

    return MakeOP(_fun)(x, y)


# =====================================================================
# TRANSPOSE
# =====================================================================

def transpose(x: Array, axes=None):
    """
    Permute tensor axes.

    Args:
        x (Array): Input tensor.
        axes (tuple[int] | None): Axis permutation.

    Returns:
        A: Transposed tensor.
    """
    lib = xp()

    def _fun(x):
        from ..array import as_nd
        from . import transpose

        out = as_nd(lib.transpose(unwrap(x), axes=axes))

        def grad_fn(g):
            if axes is None:
                rev_axes = None
            else:
                rev_axes = tuple(lib.argsort(lib.array(axes)))
            return transpose(g, axes=rev_axes),

        return out, (as_nd(x),), grad_fn

    return MakeOP(_fun)(x)


# =====================================================================
# MATMUL
# =====================================================================

def matmul(a: Array, b: Array):
    """
    Matrix multiplication: ``a @ b``.

    Supports:
        • Vector @ Vector (dot)
        • Matrix @ Vector
        • Vector @ Matrix
        • Matrix @ Matrix
        • Batched matmul (… × M × K @ … × K × N)

    Args:
        a (Array): Left operand.
        b (Array): Right operand.

    Returns:
        A: The matrix product.
    """
    lib = xp()

    def _fun(a, b):
        from ..array import as_nd
        from . import matmul
        from .primitive_array_ops import expand_dims

        out = as_nd(lib.matmul(unwrap(a), unwrap(b)))

        def grad_fn(g):
            A, B, G = a, b, g

            # ----------------------------
            # dA
            # ----------------------------
            if A.ndim == 1:              # vector @ matrix
                A2 = expand_dims(A, 0)  # (1, K)
                G2 = G if G.ndim > 1 else expand_dims(G, 0)
                dA = squeeze(matmul(G2, lib.swapaxes(unwrap(B), -1, -2)), 0)
            else:
                dA = G @ lib.swapaxes(unwrap(B), -1, -2)

            # ----------------------------
            # dB
            # ----------------------------
            if B.ndim == 1:              # matrix @ vector
                B2 = expand_dims(B, -1)  # (K, 1)
                G2 = G if G.ndim > 1 else expand_dims(G, -1)
                dB = squeeze(matmul(lib.swapaxes(unwrap(A), -1, -2), G2), -1)
            else:
                dB = matmul(lib.swapaxes(unwrap(A), -1, -2), G)

            return as_nd(dA), as_nd(dB)

        return out, (as_nd(a), as_nd(b)), grad_fn

    return MakeOP(_fun)(a, b)
