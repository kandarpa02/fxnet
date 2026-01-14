from .._typing import Array
from typing import Union, Any
from ..DType import DType, normalize_dtype
from ..array import NDarray
from ..tree_util import flatten_pytree, unflatten_pytree, map
from ...nn.base import Cell



def astype(x:Any, dtype:Union[DType, str, None]=None):
    _dt = normalize_dtype(dtype)
    return map(lambda x: NDarray(x, _dt), x)
