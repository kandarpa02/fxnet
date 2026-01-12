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


def active_tape():
    return TAPE_STACK[-1] if TAPE_STACK else None

@contextmanager
def tape():
    """
    Context manager enabling dynamic autodiff tracing.

    Usage:
        with tape():
            y = f(x)
        # The tape now contains a linearized list of Node(...)

    Behavior:
        – Pushes a new empty list into TAPE_STACK.
        – All operations wrapped with `function` append Nodes here.
        – Does NOT perform backprop. Users must call a separate backward runner.

    This design matches Chainer's "define-by-run" tape philosophy.
    """

    TAPE_STACK.append([])  
    try:
        yield
    finally:
        pass   

@contextmanager
def no_record():
    """
    Temporarily disable gradient recording for the enclosed block.

    Useful for:
      - running user code during gradient computation without re-tracing
      - avoiding infinite recursion inside grad_fns
      - cheap non-differentiable ops

    Restores previous recording state afterward.
    """

    global _RECORDING
    prev = _RECORDING
    _RECORDING = False
    try:
        yield
    finally:
        _RECORDING = prev

@contextmanager
def tracing():
    global JIT
    prev = JIT
    JIT = True
    try:
        yield
    finally:
        JIT = prev

@dataclass
class Node:
    """
    One entry in the tape, representing a single primitive operation.

    Attributes:
        out (Array):
            The forward output of the primitive.
        parents (tuple[Array]):
            The inputs to this primitive that require gradients.
        grad_fn (Callable):
            A function g -> tuple[grad_i], computing local backward.

            grad_fn(g):
                g : gradient flowing from downstream (same shape as `out`)
                returns: tuple of gradient contributions for each parent,
                         matching `parents` order.
    """

    out: Array
    parents: tuple
    grad_fn: Callable


class MakeOP:
    """
    Convert a low-level primitive into a traceable op in the autodiff system.

    A primitive must return either:
        (out, parents, grad_fn)   – explicit parent list
    or:
        (out, grad_fn)            – parents inferred as `args`

    Parameters:
        fun: Callable
            A Python function implementing the forward op. It should NOT
            mutate inputs and must return a tuple as above.

    Behavior:
        – Temporarily disables recording during `fun(*args)` execution.
          (prevents accidental double tracing)
        – Validates output format.
        – After forward pass, records Node(out, parents, grad_fn)
          into the active tape *if* recording is enabled.

    Example primitive:
        def add_fun(x, y):
            out = x + y
            def grad_fn(g):
                return g, g
            return out, (x, y), grad_fn

        add = function(add_fun)

    Calling:
        z = add(x, y)

    """

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

        # if JIT:
        #     # assign symbols
        #     for a in parents:
        #         ensure_symbol(a)

        #     out.symbol = Symbol(
        #         shape=out.np.shape,
        #         dtype=out.np.dtype
        #     )

        #     # store symbolic node
        #     t = active_tape()
        #     if t is not None:
        #         t.append(
        #             JITNode(
        #                 out=out.symbol,
        #                 parents=tuple(p.symbol for p in parents),
        #                 prim=self.fun
        #             )
        #         )


        return out


# TRACE_STACK = []

# @contextmanager
# def tracing():
#     global JIT
#     prev = JIT
#     JIT = True
#     TRACE_STACK.append([])
#     try:
#         yield
#     finally:
#         TRACE_STACK.pop()
#         JIT = prev

# class Primitive:
#     def __init__(self, name, forward, backward):
#         self.name = name
#         self.forward = forward     
#         self.backward = backward   


# class Symbol:
#     _counter = 0

#     def __init__(self, prim, inputs):
#         self.id = Symbol._counter
#         Symbol._counter += 1

#         self.prim = prim
#         self.inputs = inputs   # symbols or placeholders
#         self.shape = None
#         self.dtype = None

# class Placeholder:
#     def __init__(self, index):
#         self.index = index


# def apply_primitive(p: Primitive, *args):
#     from .array import NDarray
#     from ..backend.backend import xp

#     if JIT:
#         sym = Symbol(p, args)
#         TRACE_STACK[-1].append(sym)
#         return sym

#     else:
#         # eager
#         lib = xp()
#         xs = [a.np if isinstance(a, NDarray) else a for a in args]
#         out_np = p.forward(*xs)
#         out = NDarray(out_np)

#         if active_tape() is not None:
#             def grad_fn(g):
#                 return p.backward(g, *xs)
#             active_tape().append(Node(out, args, grad_fn))

#         return out

# class StaticGraph:
#     def __init__(self, symbols, inputs, output):
#         self.symbols = symbols
#         self.inputs = inputs      # placeholders
#         self.output = output      # symbol

# def execute_graph(graph: StaticGraph, args):
#     from .array import NDarray
#     from ..backend.backend import xp

#     env = {}

#     # bind inputs
#     for ph, arg in zip(graph.inputs, args):
#         env[ph.index] = arg.np if isinstance(arg, NDarray) else arg

#     # execute symbols
#     for sym in graph.symbols:
#         xs = []
#         for inp in sym.inputs:
#             if isinstance(inp, Placeholder):
#                 xs.append(env[inp.index])
#             else:
#                 xs.append(env[inp.id])

#         env[sym.id] = sym.prim.forward(*xs)

#     return NDarray(env[graph.output.id])

# class defgraph:
#     def __init__(self, fn):
#         self.fn = fn
#         self.graph = None

#     def __call__(self, *args):
#         from .array import NDarray

#         # First call → trace
#         if self.graph is None:
#             Symbol._counter = 0

#             placeholders = [Placeholder(i) for i in range(len(args))]

#             with tracing():
#                 out = self.fn(*placeholders)
#                 symbols = TRACE_STACK[-1]

#             self.graph = StaticGraph(
#                 symbols=symbols,
#                 inputs=placeholders,
#                 output=out
#             )

#         # Execute static graph
#         return execute_graph(self.graph, args)

# def graph(fun):
#     return defgraph(fun)