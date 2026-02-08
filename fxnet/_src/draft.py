import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Any, Optional, Dict, Set
import torch

@dataclass
class Node:
    value: Any
    parents: List[Tuple["Node", Callable]]
    
    def __hash__(self):
        return id(self)

class Tracer:
    def __init__(self, value, node=None):
        self.value = value
        self.node = node

    def __add__(self, other):
        return add(self, other)
    def __radd__(self, other):
        return add(self, other)
    def __mul__(self, other):
        return mul(self, other)
    def __rmul__(self, other):
        return mul(self, other)

    def __repr__(self):
        return f"Tracer({self.value})"

unwrap = lambda x: getattr(x, 'value', x)

def primitive(f, vjps):
    """
    f: callable (e.g. np.add, np.multiply)
    vjps: list/tuple of VJP functions, each (g, *args) -> grad w.r.t that input
    """
    def wrapped(*args):
        vals = tuple(unwrap(arg) for arg in args)
        out = f(*vals)

        # Collect Tracer parents
        parents = []
        vjp_funcs = []

        for i, (arg, vjp_fn) in enumerate(zip(args, vjps)):
            if isinstance(arg, Tracer):
                parents.append(arg.node)
                # Store the arguments for the VJP
                def make_vjp(vjp_fn, args=args, i=i):
                    return lambda g: vjp_fn(g, *args)
                vjp_funcs.append(make_vjp(vjp_fn))

        if parents:
            node = Node(out, list(zip(parents, vjp_funcs)))
            return Tracer(out, node)
        else:
            return out

    return wrapped


# VJP functions
def add_vjp0(g, x, y):
    return g

def add_vjp1(g, x, y):
    return g

def mul_vjp0(g, x, y):
    # g * y, where y might be a tracer or a value
    return g * y

def mul_vjp1(g, x, y):
    # g * x, where x might be a tracer or a value
    return g * x

add = primitive(np.add, (add_vjp0, add_vjp1))
mul = primitive(np.multiply, (mul_vjp0, mul_vjp1))


def backward(node: Node, grad):
    # Get topological order
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for parent, _ in v.parents:
                build_topo(parent)
            topo.append(v)
    
    build_topo(node)
    
    # Initialize gradients
    grads = {node: grad}
    
    # Process in reverse topological order
    for v in reversed(topo):
        g = grads[v]
        
        for parent, vjp in v.parents:
            # Compute gradient contribution from this parent
            pg = vjp(g)
            
            # Accumulate gradient
            if parent in grads:
                # For gradient accumulation, we need to use regular addition
                # not the overloaded operator that creates new tracers
                if isinstance(grads[parent], Tracer) or isinstance(pg, Tracer):
                    # If we have tracers, we need to trace through the accumulation
                    grads[parent] = grads[parent] + pg
                else:
                    # Regular numerical accumulation
                    grads[parent] = grads[parent] + pg
            else:
                grads[parent] = pg
    
    return grads


def trace(f, args):
    """Trace a function with arguments"""
    # Convert inputs to tracers
    leaves = []
    tracers = []
    for arg in args:
        leaf = Node(arg, [])
        leaves.append(leaf)
        tracers.append(Tracer(arg, leaf))
    
    out = f(*tracers)
    return out, leaves


def grad(f):
    """Return a function that computes the gradient of f"""
    def df(*args):
        # Trace the function to build computational graph
        out, leaves = trace(f, args)
        
        # Initialize gradient
        init_grad = 1.0
        
        # Backward pass
        grads = backward(out.node, init_grad)
        
        # Extract gradients for leaves
        result_grads = []
        for leaf in leaves:
            if leaf in grads:
                grad_val = grads[leaf]
                # For first-order gradients, we want to return the value
                # not a Tracer
                if isinstance(grad_val, Tracer):
                    result_grads.append(grad_val.value)
                else:
                    result_grads.append(grad_val)
            else:
                result_grads.append(0.0)
        
        if len(result_grads) == 1:
            return result_grads[0]
        return tuple(result_grads)
    
    return df

f = lambda x: x*x*x
a = Tracer(2.)
g = grad(f)
g2 = grad(g)

print('out: ', f(a))
print('g1: ', g(a))
print('g2: ', g2(a))

# python fxnet/_src/draft.py
# out:  Tracer(8.0)
# g1:  Tracer(12.0)
# g2:  Tracer(12.0)