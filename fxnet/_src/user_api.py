from .basic_functions.non_grad_utils import zeros, ones, full, zeros_like, ones_like, full_like
from torch import Tensor
import numpy as np
from ..DType import DType, dtype_f, DTypeLike

TensorLike = Tensor|np.ndarray|int|float|complex

def constant(data:TensorLike, dtype:DTypeLike|None=None):
    from .tensor_base import Texor
    return Texor(data, dtype=dtype_f(dtype))

__all__ = [
    'zeros',
    'ones',
    'zeros_like',
    'ones_like',
    'full',
    'full_like',
    'constant'
]
