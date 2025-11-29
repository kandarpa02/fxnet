from typing import Generic, TypeVar
from ..src.array import NDarray

class Variable(NDarray):
    def __init__(self, data, dtype=None, name: str = None):
        super().__init__(data, dtype)
        self.train = True
        self.name = name 

    __module__ = "faketensor"
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
    
    def __repr__(self):
        import numpy as np
        # detect if self.np is cupy array
        xp = getattr(self.np, "__array_priority__", None)
        try:
            import cupy as cp
            is_cupy = isinstance(self.np, cp.ndarray)
        except ImportError:
            is_cupy = False

        # convert to numpy for structured printing
        arr_to_print = self.np.get() if is_cupy else self.np

        dtype_str = str(getattr(self.np, "dtype", "unknown"))
        shape_str = getattr(self.np, "shape", None)
        trainable_str = self.trainable
        name_str = self.name if self.name is not None else "Variable"

        # structured printing for small arrays, truncated for large arrays
        with np.printoptions(precision=4, threshold=10, edgeitems=3, suppress=True):
            data_str = np.array2string(arr_to_print, separator=', ')

        return (f"<Variable '{name_str}' shape={shape_str} dtype={dtype_str}, "
                f"trainable={trainable_str}, data=\n{data_str}>")
    __str__ = __repr__

    
T = TypeVar("T", bound=Variable)

class Parameter(list[T], Generic[T]):
    __module__ = "faketensor"
    __qualname__ = "Parameter"

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

