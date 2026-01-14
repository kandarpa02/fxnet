from __future__ import annotations
from typing import Protocol, TypeVar, runtime_checkable, Any, Callable, Union
from numpy import ndarray
T = TypeVar("T", bound="Array")

class DType:
    pass

@runtime_checkable
class Array(Protocol):
    """A protocol representing any array-like object that supports basic arithmetic operations.
    """

    np:Any

    @property
    def shape(self:T) -> Any: ...

    @property
    def ndim(self:T) -> Any: ...

    def __add__(self: T, other: T) -> T:
        ...
    
    def __mul__(self: T, other: T) -> T:
        ...
    
    def __sub__(self: T, other: T) -> T:
        ...

    def __truediv__(self: T, other: T) -> T:
        ...

    def __repr__(self) -> str:
        ...
