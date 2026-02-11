import torch
from ..core import primitive, fxwrap
from ..tensor_base import Texor
from ...DType import DTypeLike, dtype_f
import numpy as np
from collections.abc import Sequence

TensorLike = Texor|torch.Tensor|np.ndarray|int|float
Number = int|float|complex
ShapeLike = Sequence[int]|int

def ones_like(x:TensorLike, dtype:DTypeLike=None):
    out = fxwrap(
        lambda x: torch.ones_like(x, dtype=dtype_f(dtype))
    )(x)
    return out

def zeros_like(x:TensorLike, dtype:DTypeLike=None):
    out = fxwrap(
        lambda x: torch.zeros_like(x, dtype=dtype_f(dtype))
    )(x)
    return out

def full_like(x:TensorLike, value:Number, dtype:DTypeLike=None):
    return fxwrap(
        lambda x: torch.full_like(x, value, dtype=dtype_f(dt))
    )(x)

def ones(shape:ShapeLike, dtype:DTypeLike=None):
    out = torch.ones(size=shape, dtype=dtype_f(dtype))
    return fxwrap(lambda x:x)(out)

def zeros(shape:ShapeLike, dtype:DTypeLike=None):
    out = torch.zero(size=shape, dtype=dtype_f(dtype))
    return fxwrap(lambda x:x)(out)

def full(shape:ShapeLike, value:Number, dtype:DTypeLike=None):
    out = torch.full(size=shape, fill_value=value, dtype=dtype_f(dtype))
    return fxwrap(lambda x:x)(out)