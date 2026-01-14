from .base import Cell
from typing import Protocol, Any, Union, Sequence, Callable



class Sequential(Cell):
    """
    A container module that applies a sequence of layers or callables in order.

    `Sequential` allows you to chain multiple modules or functions
    together in a simple, functional style, similar to `torch.nn.Sequential`
    or `haiku.Sequential`.

    Example
    -------
    >>> seq = Sequential([
    ...     Linear(128),
    ...     relu,
    ...     Linear(10)
    ... ])
    
    Parameters
    ----------
    layers : Sequence[Cell | Callable]
        A sequence of modules (inheriting from `Cell`) or plain callables
        (like activation functions) to apply in order.

    Notes
    -----
    - Supports any callable that takes a single argument (the input) and
      returns the output.
    - Useful for creating MLPs or simple layer stacks with minimal boilerplate.
    """

    def __init__(self, layers:Sequence[Cell|Callable], name: str | None = None):
        super().__init__(name)
        self.layers = layers
        self.__glue_layers__()
    
    def __glue_layers__(self):
        for i, l in enumerate(self.layers):
            name = l.__class__.__name__ if isinstance(l, Cell) else l.__name__
            setattr(self, f"{name}{i}", l)

    def call(self, *args):
        for i, lay in enumerate(self.layers):
            if i == 0:
                x = lay(*args)
            else:
                x = lay(x)
        return x
    
