from .._typing import Array
from typing import Union, Any
from ..DType import DType, normalize_dtype
from ..array import NDarray
from ..tree_util import flatten_pytree, unflatten_pytree, map
from ...nn.base import Cell
from ..base import MakeOP


def astype(x:Any, dtype:Union[DType, str, None]=None):
    _dt = normalize_dtype(dtype)

    def fun(x):
        _x = map(lambda x: NDarray(x, _dt), x)
        def grad_f(g):
            # return g,
            return map(lambda x:x, g)
    
        return _x, grad_f
    return MakeOP(fun)(x)