from .base import Cell
from ..src.functions.primitive_array_ops import reshape

class Flatten(Cell):
    def __init__(self, order=1, name: str | None = None):
        super().__init__(name)
        self.order = order

    def call(self, x):
        shape = x.shape
        ndim = len(shape)

        if self.order < 0 or self.order > ndim:
            raise ValueError(
                f"Flatten order {self.order} invalid for shape {shape}"
            )

        if self.order == ndim:
            return x

        left = shape[:self.order]
        return reshape(x, [*left, -1])
