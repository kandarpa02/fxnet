from ._typing import Array
import numpy as np
from .functions import (
    add,
    multiply,
    subtract,
    negative,
    divide,
    power
)

def as_ndarray(x):
    if isinstance(x, (np.ndarray, int, float, bool, list)):
        return np.asarray(x)
    elif isinstance(x, NDarray):
        return x.np
    else:
        raise TypeError(f"{type(x)} is not supported as input")

def as_nd(x):
    return NDarray(x)

class NDarray(Array):
    def __init__(self, data, dtype=None) -> None:
        super().__init__()
        self.np = as_ndarray(data).astype(dtype) if dtype else as_ndarray(data)
    
    # -------------------------
    # Basic attributes
    # -------------------------
    @property
    def dtype(self):
        return self.np.dtype.__str__()

    @property
    def shape(self):
        return self.np.shape

    def __len__(self):
        return len(self.np)

    def astype(self, dtype):
        """Return a new NDarray with the same values, different dtype."""
        return NDarray(self.np, dtype=dtype)

    # -------------------------
    # Display helpers
    # -------------------------
    def __repr__(self):
        return repr(self.np)

    def __str__(self):
        return str(self.np)

    def __array__(self):
        """Allows NumPy to extract underlying data when needed."""
        return self.np

    __array_priority__ = 200  # ensure our ops dominate numpy’s

    def __float__(self):
        return float(self.np)
    
    def __int__(self):
        return float(self.np)

    # -------------------------
    # Unary ops
    # -------------------------
    def __neg__(self):
        return negative(self)

    # -------------------------
    # Binary ops (forward)
    # -------------------------
    def __add__(self, other):
        other = as_nd(other)
        return add(self, other)

    def __sub__(self, other):
        other = as_nd(other)
        return subtract(self, other)

    def __mul__(self, other):
        other = as_nd(other)
        return as_nd(multiply(self, other))

    def __truediv__(self, other):
        other = as_nd(other)
        return divide(self, other)
    
    def __pow__(self, other):
        other = as_nd(other)
        return power(self, other)

    # -------------------------
    # Binary ops (reverse)
    # -------------------------
    def __radd__(self, other):
        other = as_nd(other)
        return add(other, self)

    def __rsub__(self, other):
        other = as_nd(other)
        return subtract(other, self)

    def __rmul__(self, other):
        other = as_nd(other)
        return multiply(other, self)

    def __rtruediv__(self, other):
        other = as_nd(other)
        return divide(other, self)
    
    def __rpow__(self, other):
        other = as_nd(other)
        return power(self, other)
