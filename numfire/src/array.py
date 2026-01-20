from ._typing import Array as A
from ..backend.backend import xp   
from .functions import *
from .functions.comparison import (
    equal, not_equal, 
    greater, greater_equal, 
    less, less_equal, 
    logical_not, logical_and, 
    logical_or, logical_xor, 
    logical_all, logical_any
    )

from typing import Optional
from typing import Union, NamedTuple
from .DType import DType, normalize_dtype
from ..src.functions.xpy_utils import get_dev, module

# -------------------------
# Backend-aware array casting
# -------------------------

import string
import random

NAME_COUNTER = 0

ASCII = string.ascii_letters
DIGITS = string.digits
EXTRAS = "@#%$?"
POOL = ASCII + DIGITS + EXTRAS

def name(length=20):
    global NAME_COUNTER

    rng = random.Random(NAME_COUNTER)
    out = ''.join(rng.choice(POOL) for _ in range(length))

    NAME_COUNTER += 1
    return out

_Dtype = Union[DType, str, None]

def as_ndarray(x):
    """
    Normalize input to a backend ndarray WITHOUT changing device.
    """
    import numpy as np
    try:
        import cupy as cp
        has_cupy = True
    except ImportError:
        cp = None
        has_cupy = False

    # ---- Backend ndarrays (KEEP DEVICE) ----
    if has_cupy and isinstance(x, cp.ndarray):
        return x

    if isinstance(x, np.ndarray):
        return x

    # ---- Our NDarray (unwrap only) ----
    if isinstance(x, NDarray):
        return x.__backend_buffer__

    # ---- NumPy scalar ----
    if isinstance(x, np.generic):
        return np.asarray(x)

    # ---- CuPy scalar ----
    if has_cupy and isinstance(x, cp.generic):
        return cp.asarray(x)

    # ---- Python scalars / lists ----
    if isinstance(x, (int, float, bool, list, tuple)):
        if has_cupy:
            return cp.asarray(x)
        return np.asarray(x)

    raise TypeError(f"{type(x)} not supported as input")


def as_nd(x):
    return NDarray(x)

class _AtIndexer:
    def __init__(self, x):
        self.x = x

    def __getitem__(self, idx):
        return _AtSet(self.x, idx)


class _AtSet:
    def __init__(self, x, idx):
        self.x = x
        self.idx = idx

    def set(self, value):
        new = self.x.__backend_buffer__.copy()
        new[self.idx] = value
        return NDarray(new)

# -------------------------
# NDarray class
# -------------------------

class NDarray(A):
    def __init__(self, data, dtype=None) -> None:
        super().__init__()
        arr = as_ndarray(data)
        self.__backend_buffer__ = arr.astype(dtype) if dtype else arr
        self.train = True
        self.id = name()
        
    __is_leaf__ = True
    __module__ = "numfire"
    __qualname__ = "NDarray"

    @property
    def np(self):
        arr = self.__backend_buffer__.view()
        arr.flags.writeable = False
        return arr

    @property
    def trainable(self):
        return self.train
    
    @property
    def dtype(self):
        return self.__backend_buffer__.dtype.__str__()

    @property
    def shape(self):
        return self.__backend_buffer__.shape
    
    @property
    def ndim(self):
        return self.__backend_buffer__.ndim
    
    @property
    def size(self):
        return self.__backend_buffer__.size

    def __len__(self):
        return len(self.__backend_buffer__)
    
    def __hash__(self):
        return self.id   # identity-based hashing

    # -------------------------
    # Display helpers
    # -------------------------
    def __repr__(self):
        return repr(self.__backend_buffer__)

    def __str__(self):
        return str(self.__backend_buffer__)

    def __array__(self):
        """Allows NumPy to extract underlying data when needed."""
        return self.__backend_buffer__

    @property
    def __cuda_array_interface__(self):
        return self.__backend_buffer__.__cuda_array_interface__

    __array_priority__ = 200 

    def __mutate_state__(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dict__.keys():

                setattr(self, k, v)

    def __without_mutation__(self, **kwargs):
        pass

    def copy(self):
        x = self.__backend_buffer__.copy()
        copied = NDarray(x)
        copied.__mutate_state__(id=self.id)
        return copied

    def astype(self, dtype:_Dtype):
        from ..src.ndarray.utils import astype
        return astype(self, dtype=dtype)
    
    def __float__(self):
        return float(self.__backend_buffer__)

    def __int__(self):
        return int(self.__backend_buffer__)
    
    def __setitem__(self, k, v):
        self.__backend_buffer__[k] = v

    def __getitem__(self, idx):
        return NDarray(self.__backend_buffer__[idx].copy())
    
    @property
    def at(self):
        return _AtIndexer(self)

    # -------------------------
    # Array makers
    # -------------------------

    def full_like(self, val, dtype=None):
        return NDarray(xp().full_like(self.__backend_buffer__, val, dtype=dtype))

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
        return add(self, as_nd(other).astype(self.dtype))
        # return add(self, as_nd(other))

    def __sub__(self, other):
        return subtract(self, as_nd(other).astype(self.dtype))
        # return subtract(self, as_nd(other))

    def __mul__(self, other):
        return multiply(self, as_nd(other).astype(self.dtype))
        # return multiply(self, as_nd(other))

    def __truediv__(self, other):
        return divide(self, as_nd(other).astype(self.dtype))

    def __pow__(self, other):
        return power(self, as_nd(other).astype(self.dtype))
    
    def __matmul__(self, other):
        return matmul(self, other)
    
    def __and__(self, other):
        return logical_and(self, other)

    def __or__(self, other):
        return logical_or(self, other)

    def __xor__(self, other):
        return logical_xor(self, other)

    def __invert__(self):
        return logical_not(self)
    
    def any(self, axis=None, keepdims=False):
        return logical_any(self, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False):
        return logical_all(self, axis=axis, keepdims=keepdims)

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

