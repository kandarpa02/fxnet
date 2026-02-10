import torch
from ..core import primitive, fxwrap
from ..tensor_base import Texor
from ...DType import DTypeLike, dtype_f

@fxwrap
def ones_like(x:Texor, dtype:DTypeLike):
    return torch.ones_like(x, dtype=dtype_f(dtype))

@fxwrap
def zeros_like(x:Texor, dtype:DTypeLike):
    return torch.zeros_like(x, dtype=dtype_f(dtype))

