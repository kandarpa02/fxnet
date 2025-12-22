from .._typing import Array
from typing import Union
from ..DType import DType, normalize_dtype
from ..array import NDarray

def astype(x:Array, dtype:Union[DType, str, None]=None):
    _dt = normalize_dtype(dtype)
    return NDarray(x, _dt)