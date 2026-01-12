from ._typing import Array as A
from ..backend.backend import xp    # unified backend (numpy OR cupy)
from .functions import *
from .functions.comparison import equal, not_equal, greater, greater_equal, less, less_equal, logical_not
from typing import Optional
from typing import Union
from .DType import DType
# -------------------------
# Backend-aware array casting
# -------------------------

_Dtype = Union[DType, str, None]

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

class NDarray(A):
    def __init__(self, data, dtype=None) -> None:
        super().__init__()
        arr = as_ndarray(data)
        self.np = arr.astype(dtype) if dtype else arr
        self.train = True
        self.symbol = None
        
    __is_leaf__ = True
    __module__ = "mathino"
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
    
    @property
    def size(self):
        return self.np.size

    def __len__(self):
        return len(self.np)

    def astype(self, dtype:_Dtype):
        from ..src.ndarray.utils import astype
        """Return a new NDarray with the same values, different dtype."""
        return astype(self, dtype)
    
    def __hash__(self):
        return id(self)   # identity-based hashing

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

    @property
    def __cuda_array_interface__(self):
        return self.np.__cuda_array_interface__

    __array_priority__ = 200 

    def __float__(self):
        return float(self.np)

    def __int__(self):
        return int(self.np)
    
    def __setitem__(self, k, v):
        self.np[k] = v

    def __getitem__(self, id):
        return NDarray(self.np[id])
    
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
    
    def __invert__(self):
        return logical_not(self)

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

    def __eq__(self, other):
        return equal(self, as_nd(other))

    def __ne__(self, other):
        return not_equal(self, as_nd(other))

    def __lt__(self, other):
        return less(self, as_nd(other))

    def __le__(self, other):
        return less_equal(self, as_nd(other))

    def __gt__(self, other):
        return greater(self, as_nd(other))

    def __ge__(self, other):
        return greater_equal(self, as_nd(other))
    
    def __req__(self, other):
        return equal(as_nd(other), self)

    def __rne__(self, other):
        return not_equal(as_nd(other), self)

    def __rlt__(self, other):
        return less(as_nd(other), self)

    def __rle__(self, other):
        return less_equal(as_nd(other), self)

    def __rgt__(self, other):
        return greater(as_nd(other), self)

    def __rge__(self, other):
        return greater_equal(as_nd(other), self)

