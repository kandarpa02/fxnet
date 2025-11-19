from ._typing import arraytype
import numpy as np
from .functions import add, mul

def as_ndarray(x):
    if isinstance(x, np.ndarray|int|float|bool|list):
        return np.asarray(x)
    elif isinstance(x, NDarray):
        return x.np
    else:
        raise TypeError(f"{type(x)} is not supported as input")

class NDarray(arraytype):
    def __init__(self, data) -> None:
        super().__init__()
        self.np = as_ndarray(data)

    @property
    def shape(self):
        return self.np.shape

    def __repr__(self):
        return self.np.__repr__()
    
    def __str__(self) -> str:
        return self.np.__str__()
    
    def __array__(self):
        return self.np

    __array_priority__ = 200

    def __add__(self, other):
        return add(self, other)
    
    def __mul__(self, other):
        return mul(self, other)