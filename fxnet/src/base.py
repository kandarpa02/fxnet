"""
Base utilities for FakeTensor's functional automatic differentiation system.

This module implements:
  • A global tape stack for dynamic/eager-mode autodiff.
  • A `function` wrapper that turns a primitive into a traceable op.
  • Context managers controlling whether operations are recorded.
  • The `Node` structure storing parents + backward function.

FakeTensor follows a *functional* autograd design:
  – Arrays do NOT store gradient or graph info.
  – Each primitive returns `(out, parents, grad_fn)` describing the local rule.
  – Tapes collect nodes dynamically (similar to Chainer).
  – Backprop is implemented externally by reading the tape.

This file contains no gradient logic; it only defines how ops are logged.
"""


from typing import List, Callable, Protocol, Union
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
from .utils import broadcast_backward
from ._typing import Array 

# When True, primitives executed inside a `tape()` block
# append a Node(out, parents, grad_fn) into the active tape.
_RECORDING = True
JIT = False

# The tape stack enables nested tapes.
# Each active tape is a list of Node objects.
TAPE_STACK = []
JIT_STACK = None

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

    out: Array
    parents: tuple
    grad_fn: Callable


class MakeOP:
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
                    f"Function '{self.fun.__name__}' must return a tuple"
                )

            n = len(output)

            if n == 3:
                out, parents, grad_fn = output
                if not isinstance(parents, (tuple, list)):
                    raise TypeError("parents must be tuple/list")

            elif n == 2:
                out, grad_fn = output
                parents = args

            else:
                raise ValueError("Function must return (out, parents, grad_fn) or (out, grad_fn)")
            
            if not callable(grad_fn):
                raise TypeError("grad_fn must be callable.")

        finally:
            _RECORDING = prev

        # append to tape for dynamic mode
        t = active_tape()
        if t is not None and _RECORDING:
            t.append(Node(out, parents, grad_fn))

        return out


@contextmanager
def tracing():
    global JIT
    prev = JIT
    JIT = True
    try:
        yield

    finally:
        JIT = prev
