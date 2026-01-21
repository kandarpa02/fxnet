from typing import Generic, TypeVar
from ..src.array import NDarray
from ..src.DType import DType, normalize_dtype
def as_var(data):
    return getattr(data, "__backend_buffer__", data)

def _check(data):
    if isinstance(data, Variable|NDarray):
        return data.__backend_buffer__
    else:
        return data

class Variable(NDarray):
    def __init__(self, data, dtype:DType|None=None, name: str|None = None):
        super().__init__(as_var(data), normalize_dtype(dtype))
        self.train = True
        self.name = name if name is not None else 'Variable'

    __module__ = "numfire.nn"
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

    def to_ndarray(self):
        return NDarray(self.__backend_buffer__.copy())
    
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
        return self._repr()

    __str__ = __repr__

T = TypeVar("T", bound=Variable|NDarray)

class Parameter(list[T], Generic[T]):
    __module__ = "numfire.nn"
    __qualname__ = "Parameter"

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

