from dataclasses import dataclass
from typing import Callable, Any
from contextlib import contextmanager
from .utils import Meta

TRACING = False
JIT_STACK = []

@contextmanager
def trace_mode():
    JIT_STACK.append([])  
    try:
        yield
    finally:
        pass   

def element_wise(a, b, name, func):
    shape = Meta.element_wise_shape(a, b)
    dtype = Meta.DType(a, b)
    out = FT_Tracer(
        shape,
        dtype,
        name=name,
        func=func,
        parents=(a, b)
    )
    return out

def vectorize(a, b, name, func):
    shape = Meta.dot_shape(a, b)
    dtype = Meta.DType(a, b)
    out = FT_Tracer(
        shape,
        dtype,
        name=name,
        func=func,
        parents=(a, b)
    )
    return out

class FT_Tracer:
    def __init__(self, shape:tuple, dtype:str, name:str, func:Callable=lambda:None, parents:tuple=()) -> None:
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.func = func
        self.parents = parents

    def __repr__(self):
        return f"FT_Tracer(shape={self.shape}, dtype='{self.dtype}', name='{self.name}')"
    
    def __str__(self): return self.__repr__()

    def __add__(self, other):
        return element_wise(self, other, 'add', lambda a, b: a + b)
    
    def __mul__(self, other):
        return element_wise(self, other, 'mul', lambda a, b: a * b)

    def __sub__(self, other):
        return element_wise(self, other, 'sub', lambda a, b: a - b)
    
    def __truediv__(self, other):
        return element_wise(self, other, 'div', lambda a, b: a / b)
    
    # def __matmul__(self, other):
    #     return vectorize(self, other, 'matmul', )

    