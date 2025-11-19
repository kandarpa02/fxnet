from typing import List, Callable, Protocol, Union
from contextlib import contextmanager

from dataclasses import dataclass
import numpy as np
from .utils import broadcast_backward

from ._typing import arraytype 

_RECORDING = True

TAPE_STACK = []

def active_tape():
    return TAPE_STACK[-1] if TAPE_STACK else None

@contextmanager
def tape():
    TAPE_STACK.append([])  
    try:
        yield
    finally:
        pass   

@contextmanager
def no_record():
    global _RECORDING
    prev = _RECORDING
    _RECORDING = False
    try:
        yield
    finally:
        _RECORDING = prev

@dataclass
class Node:
    out: np.ndarray
    parents: tuple
    grad_fn: Callable


class function:
    def __init__(self, fun):
        self.fun = fun

    def __call__(self, *args):
        global _RECORDING

        prev = _RECORDING
        _RECORDING = False
        try:
            output = self.fun(*args)

            if not isinstance(output, tuple):
                raise TypeError(
                    f"Function '{self.fun.__name__}' must return a tuple, "
                    f"got {type(output).__name__}"
                )

            n = len(output)

            if n == 3:
                out, parents, grad_fn = output

                # validate parents
                if not isinstance(parents, (tuple, list)):
                    raise TypeError(
                        f"Expected parents to be tuple/list, got {type(parents).__name__}"
                    )

            elif n == 2:
                out, grad_fn = output
                parents = args

            else:
                raise ValueError(
                    f"Function '{self.fun.__name__}' must return either "
                    f"(out, grad_fn) or (out, parents, grad_fn); got {n} values"
                )

            if not callable(grad_fn):
                raise TypeError(
                    f"grad_fn must be callable, got {type(grad_fn).__name__}"
                )

        finally:
            # Always restore recording state
            _RECORDING = prev

        t = active_tape()
        if t is not None and _RECORDING:
            t.append(Node(out, parents, grad_fn))

        return out
