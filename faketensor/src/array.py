from ._typing import Array
from ..backend.backend import xp    # unified backend (numpy OR cupy)
from .functions import *
from typing import Optional

# -------------------------
# Backend-aware array casting
# -------------------------

def as_ndarray(x):
    """
    Convert input to backend array (numpy OR cupy) while respecting NDarray.
    """
    lib = xp()

    # If it's already a backend ndarray
    if isinstance(x, lib.ndarray):
        return x

    # Python scalars
    if isinstance(x, (int, float, bool)):
        return lib.asarray(x)

    # Lists or tuples
    if isinstance(x, (list, tuple)):
        return lib.asarray(x)

    # If user passed a raw numpy array → convert to backend array
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return lib.asarray(x)
    except Exception:
        pass

    # Our NDarray
    if isinstance(x, NDarray):
        return lib.asarray(x.np)
    
    if lib.isscalar(x):            
        return lib.array(x)

    raise TypeError(f"{type(x)} not supported as input")


def as_nd(x):
    return NDarray(x)


# -------------------------
# NDarray class
# -------------------------

class NDarray(Array):
    def __init__(self, data, dtype=None) -> None:
        super().__init__()
        arr = as_ndarray(data)
        self.np = arr.astype(dtype) if dtype else arr
        self.train = False
    __is_leaf__ = True
    __module__ = "faketensor"
    __qualname__ = "NDarray"
    # -------------------------
    # Basic attributes
    # -------------------------
    @property
    def trainable(self):
        return self.train
    
    @property
    def dtype(self):
        return self.np.dtype.__str__()

    @property
    def shape(self):
        return self.np.shape
    
    @property
    def ndim(self):
        return self.np.ndim

    def __len__(self):
        return len(self.np)

    def astype(self, dtype):
        """Return a new NDarray with the same values, different dtype."""
        return NDarray(self.np, dtype=dtype)
    
    def __hash__(self):
        return id(self)   # identity-based hashing

    def __eq__(self, other):
        return self is other

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
        return int(self.np)
    
    # -------------------------
    # Array makers
    # -------------------------

    def full_like(self, val, dtype=None):
        return NDarray(xp().full_like(self.np, val, dtype=dtype))

    # -------------------------
    # Unary ops
    # -------------------------
    def __neg__(self):
        from ..src.functions import negative
        return negative(self)

    # -------------------------
    # Binary ops (forward)
    # -------------------------
    def __add__(self, other):
        return add(self, as_nd(other))

    def __sub__(self, other):
        return subtract(self, as_nd(other))

    def __mul__(self, other):
        return multiply(self, as_nd(other))

    def __truediv__(self, other):
        return divide(self, as_nd(other))

    def __pow__(self, other):
        return power(self, as_nd(other))
    
    def __matmul__(self, other):
        return matmul(self, other)

    @property
    def T(self):
        return transpose(self)
    # -------------------------
    # Binary ops (reverse)
    # -------------------------
    def __radd__(self, other):
        return add(as_nd(other), self)

    def __rsub__(self, other):
        return subtract(as_nd(other), self)

    def __rmul__(self, other):
        return multiply(as_nd(other), self)

    def __rtruediv__(self, other):
        return divide(as_nd(other), self)

    def __rpow__(self, other):
        return power(as_nd(other), self)
