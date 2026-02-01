from ._typing import Array as A
from ..backend.backend import xp   
import numpy as np
import torch
from torch import tensor, dtype
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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if isinstance(x, np.ndarray):
        return tensor(x, device=device).detach()

    if isinstance(x, NDarray):
        return x.__backend_buffer__

    if isinstance(x, np.generic):
        return tensor(x, device=device).detach()

    if isinstance(x, (int, float, bool)):
        return tensor(x, device=device).detach()
    
    if isinstance(x, (list, tuple)):
        if any(isinstance(_, NDarray) for _ in x):
            return tensor(tuple(_.__backend_buffer__ for _ in x), device=device).detach()
        return tensor(x, device=device).detach()
    
    if isinstance(x, torch.Tensor):
        return x

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
    def __init__(self, data, dtype:dtype|None=None) -> None:
        super().__init__()
        arr = as_ndarray(data)
        self.__backend_buffer__ = arr.to(dtype=dtype) if dtype else arr
        self.train = True
        self.id = name()
        
    __is_leaf__ = True
    __module__ = "numfire"
    __qualname__ = "NDarray"

    @property
    def np(self):
        arr = self.__backend_buffer__
        # arr.flags.writeable = False
        return arr

    @property
    def trainable(self):
        return self.train
    
    @property
    def dtype(self):
        _str = self.__backend_buffer__.dtype.__str__()
        return DType.from_torch_dtype(_str)

    @property
    def shape(self):
        return tuple(self.__backend_buffer__.shape)
    
    @property
    def ndim(self):
        return self.__backend_buffer__.ndim
    
    @property
    def size(self):
        return self.__backend_buffer__.numel()

    def __len__(self):
        return len(self.__backend_buffer__)
    
    def __hash__(self):
        return self.id   # identity-based hashing

    # -------------------------
    # Display helpers
    # -------------------------
    def __repr__(self):
        return 'Tensor'+self.__backend_buffer__.__str__().removeprefix('tensor')

    def __str__(self):
        return self.__repr__()

    def __array__(self):
        """Allows NumPy to extract underlying data when needed."""
        return self.__backend_buffer__
    
    def __tensor__(self):
        return self.__backend_buffer__

    # @property
    # def __cuda_array_interface__(self):
    #     return self.__backend_buffer__.__cuda_array_interface__

    __array_priority__ = 200 

    def __mutate_state__(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dict__.keys():

                setattr(self, k, v)

    def __without_mutation__(self, **kwargs):
        pass

    def copy(self):
        x = torch.t_copy(self.__backend_buffer__)
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
        return NDarray(self.__backend_buffer__[idx])
    
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


from typing import Generic, TypeVar
def as_var(data):
    return getattr(data, "__backend_buffer__", data)

def _check(data):
    if isinstance(data, Variable|NDarray):
        return data.__backend_buffer__
    else:
        return data

class Variable(NDarray):
    """
    Create a mutable, trainable tensor with an explicit identity.

    `Variable` represents model parameters or any value that is expected
    to be updated during optimization. Unlike `constant`, a Variable
    participates in gradient computation and can be modified by
    optimizers.

    Each Variable has a stable identity and optional name, making it
    suitable for parameter tracking, checkpointing, and debugging.

    Parameters
    ----------
    x : TensorLike
        Initial value of the variable. Can be a Python scalar, list,
        tuple, NumPy array, or compatible tensor-like object.
    dtype : DtypeLike
        Data type of the variable.
    name : str, optional
        Human-readable identifier for the variable. Used for debugging,
        parameter inspection, and serialization.
    trainable : bool, default=True
        Whether the variable participates in gradient-based optimization.

    Attributes
    ----------
    value : Tensor
        The underlying tensor value.
    name : str or None
        Name of the variable.
    trainable : bool
        Whether gradients are accumulated and applied to this variable.

    Examples
    --------
    >>> a = nf.Variable([1., 3, 5, 6], nf.float32, name="a")
    >>> print(a)
    Variable('a', shape=(4,), dtype=numfire.float32, trainable=True)
    Tensor([1., 3., 5., 6.])
    """

    def __init__(self, data, dtype:DType|str|None=None, name: str|None = None):
        super().__init__(as_var(data), normalize_dtype(dtype))
        self.train = True
        self.name = name if name is not None else 'Variable'

    __module__ = "numfire."
    __qualname__ = "Variable"

    @property
    def np(self):
        return self.__backend_buffer__

    @property
    def trainable(self):
        return self.train
    
    def freeze(self):
        self.train = False

    def unfreeze(self):
        self.train = True
    
    def assign(self, value):
        value = _check(value)
        self.__backend_buffer__[...] = value

    def _repr(self):
        name, shape_str, dtype_str, trainable_str = self.name, self.shape, self.dtype, self.trainable
        indent = len("Variable(") * " "
        pref = f"Variable('{name}', shape={shape_str} dtype={dtype_str}, trainable={trainable_str}\n"
        data = self.np
        pref += data.__repr__()
        return pref + ")"
    
    def __repr__(self):
        name, shape_str, dtype_str, trainable_str = self.name, self.shape, self.dtype, self.trainable
        # indent = len("Variable(") * " "
        pref = f"Variable('{name}', shape={shape_str} dtype={dtype_str}, trainable={trainable_str}\n"
        return pref + 'Tensor'+self.__backend_buffer__.__str__().removeprefix('tensor') + ")"

    __str__ = __repr__

T = TypeVar("T", bound=Variable|NDarray)

class Parameter(list[T], Generic[T]):
    __module__ = "numfire.nn"
    __qualname__ = "Parameter"

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

