from __future__ import annotations
from typing import Protocol, TypeVar, runtime_checkable, Any, Callable

T = TypeVar("T", bound="Tensor")


@runtime_checkable
class Tensor(Protocol):
    """A protocol representing any array-like object that supports basic arithmetic operations.

    This serves as a base interface for all array types in the static graph system,
    such as `StaticArray`, `DynamicArray`, or GPU-backed arrays.

    Any class implementing this protocol must define at least:
        - `__add__`
        - `__repr__`
        - A data storage attribute (like `_storage`)

    Example:
        ```python
        class StaticArray(Tensor):
            def __init__(self, data):
                self._storage = np.array(data)

            def __add__(self, other):
                return StaticArray(self._storage + other._storage)

            def __repr__(self):
                return f"Array({self._storage})"
        ```
    """

    _storage: Any

    def __add__(self: T, other: T) -> T:
        ...

    def __repr__(self) -> str:
        ...