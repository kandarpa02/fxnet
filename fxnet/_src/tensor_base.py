import torch
import numpy as np
from .basic_functions.vjps import *
from .basic_functions.vjps import _getitem
from ..DType import DType, dtype_f, DTypeLike

ONLY = {
    # --- Python internals (do NOT touch) ---
    "__class__", "__dict__", "__repr__", "__str__", "__hash__",
    "__getattribute__", "__setattr__", "__delattr__",
    "__init__", "__new__", "__torch_function__", "_node",

    # --- minimal tensor inspection ---
    "shape", "dtype", "device", "ndim", "numel",

    # --- your helpers ---
    "size", "astype", "_tensor",

    "__add__", "__radd__",
    "__sub__", "__rsub__",
    "__mul__", "__rmul__",
    "__truediv__", "__rtruediv__",
    "__neg__",
    "__pow__", "__rpow__",
    "__matmul__", "__rmatmul__",
    "T",

    # --- comparisons ---
    "__gt__", "__rgt__",
    "__ge__", "__rge__",
    "__lt__", "__rlt__",
    "__le__", "__rle__",
    "__eq__", "__req__",
    "__ne__", "__rne__",
    "__and__", "__rand__", "__or__", "__ror__", "__invert__", "__xor__", "__rxor__",

    # --- math methods you mapped to primitives ---
    "exp", "log", "sin", "cos", "tanh",
    "sum", "mean",
    "reshape", "view",
    "permute", "transpose",
    "squeeze", "unsqueeze",

    # --- indexing ---
    "__getitem__",

    "as_subclass", "__array_prioroty__", "__array__",
}

def data_manager(data, dtype, device):
    if isinstance(data, Texor):
        if dtype is not None:
            data = torch.as_tensor(data, dtype=dtype)
        if device is not None:
            data = torch.as_tensor(data, device=device)
        return data 
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device).detach()
    
    if isinstance(data, np.ndarray|int|float|bool|complex):
        return torch.as_tensor(data, dtype=dtype, device=device).detach()
    
    return data

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Texor(torch.Tensor):
    __qualname__ = 'Tensor'
    __module__ = 'fxnet'

    def __hash__(self):
        return id(self)

    @staticmethod
    def __new__(cls, data, dtype=None, device=None):
        dtype = dtype_f(dtype)
        global DEVICE
        device = device if device is not None else DEVICE
        data = data_manager(data, dtype, device)
        obj = torch.Tensor._make_subclass(cls, data, require_grad=False)
        obj._node = None
        return obj

    def __init__(self, data, dtype=None, device=None):
        pass

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # unwrap Texor -> Tensor
        def unwrap(x):
            return x.as_subclass(torch.Tensor) if isinstance(x, Texor) else x

        unwrapped_args = tuple(unwrap(a) for a in args)

        # run real torch op
        out = func(*unwrapped_args, **kwargs)

        # wrap outputs back to Texor
        def wrap(x):
            if isinstance(x, torch.Tensor):
                tx = Texor(x)
                return tx
            return x

        if isinstance(out, tuple):
            return tuple(wrap(o) for o in out)
        return wrap(out)
    
    def __getattribute__(self, name):
        if not name in ONLY:
            raise AttributeError(
                f"fxnet.Tensor does not expose '{name}'. Use fxnet primitives."
            )
        return super().__getattribute__(name)
    
    @property
    def _tensor(self): return torch.Tensor._make_subclass(torch.Tensor, self, False)
    
    def __repr__(self): return repr(self._tensor).replace('tensor', 'Tensor').replace('torch', 'fxnet')
    
    def __str__(self): return self.__repr__()

    def size(self): return self.numel()

    @property
    def shape(self): return tuple(self._tensor.shape)

    def astype(self, dtype:DTypeLike): return Texor(self, dtype_f(dtype))

    @property
    def dtype(self): DType(self._tensor.dtype)

    def __array__(self, dtype=None):
        arr = (
            self._tensor
            .detach()
            .cpu()
            .numpy()
        )
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    
    def __add__(self, other): return add(self, other)
    def __radd__(self, other): return add(other, self)
    def __sub__(self, other): return sub(self, other)
    def __rsub__(self, other): return sub(other, self)
    def __mul__(self, other): return mul(self, other)
    def __rmul__(self, other): return mul(other, self)
    def __truediv__(self, other): return div(self, other)
    def __rtruediv__(self, other): return div(other, self)
    def __neg__(self): return neg(self)
    def __pow__(self, other): return pow(self, other)
    def __rpow__(self, other): return pow(other, self)
    def __matmul__(self, other): return matmul(self, other)
    def __rmatmul__(self, other): return matmul(other, self)
    def exp(self): return exp(self)
    def log(self): return log(self)
    def sin(self): return sin(self)
    def cos(self): return cos(self)
    def tanh(self): return tanh(self)
    def sum(self, axis=None, keepdims=False): return sum(self, axis=None, keepdims=False)
    def mean(self, axis=None, keepdims=False): return mean(self, axis=None, keepdims=False)
    def reshape(self, *shape): return reshape(self, shape)
    def view(self, *shape): return reshape(self, *shape)
    def permute(self, *axes): return permute(self, axes)
    def transpose(self, *axes): return permute(self, axes)

    @property
    def T(self):
        axes = tuple(reversed(range(self.ndim)))
        return permute(self, axes)

    def squeeze(self, axis=None): return squeeze(self, axis)
    def unsqueeze(self, axis=None): return unsqueeze(self, axis)

    def __getitem__(self, idx): return _getitem(self, idx)
    def __gt__(self, other): return greater(self, other)
    def __rgt__(self, other): return greater(other, self)
    def __ge__(self, other): return greater_equal(self, other)
    def __rge__(self, other): return greater_equal(other, self)
    def __lt__(self, other): return less(self, other)
    def __rlt__(self, other): return less(other, self)
    def __le__(self, other): return less_equal(self, other)
    def __rle__(self, other): return less_equal(other, self)
    def __eq__(self, other): return equal(self, other)
    def __req__(self, other): return equal(other, self)
    def __ne__(self, other): return not_equal(self, other)
    def __rne__(self, other): return not_equal(other, self)
    def __and__(self, other): return logical_and(self, other)
    def __rand__(self, other): return logical_and(other, self)
    def __or__(self, other): return logical_or(self, other)
    def __ror__(self, other): return logical_or(other, self)
    def __invert__(self): return logical_not(self)
    def __xor__(self, other): return logical_xor(self, other)
    def __rxor__(self, other): return logical_xor(other, self)


