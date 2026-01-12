from typing import Generic, TypeVar
from ..src.array import NDarray

class Variable(NDarray):
    def __init__(self, data, dtype=None, name: str = None):
        super().__init__(data, dtype)
        self.train = True
        self.name = name if name is not None else 'Variable'

    __module__ = "mathino.nn"
    __qualname__ = "Variable"

    @property
    def trainable(self):
        return self.train
    
    def freeze(self):
        self.train = False

    def unfreeze(self):
        self.train = True

    def to_ndarray(self):
        return NDarray(self.np)
    
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

T = TypeVar("T", bound=Variable)

class Parameter(list[T], Generic[T]):
    __module__ = "mathino.nn"
    __qualname__ = "Parameter"

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

