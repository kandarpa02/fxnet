from .core import Node
from .tensor_base import Tracer
from ..tree_util import flatten_pytree, unflatten_pytree
from .basic_functions.non_grad_utils import ONES_LIKE
import torch

# def backward(node: Node, grad):
#     # Get topological order
#     topo = []
#     visited = set()
    
#     def build_topo(v):
#         if v not in visited:
#             visited.add(v)
#             for parent, _ in v.parents:
#                 build_topo(parent)
#             topo.append(v)
    
#     build_topo(node)
    
#     # Initialize gradients
#     grads = {node: grad}
    
#     # Process in reverse topological order
#     for v in reversed(topo):
#         g = grads[v]
        
#         for parent, vjp in v.parents:
#             # Compute gradient contribution from this parent
#             pg = vjp(g)
            
#             # Accumulate gradient
#             if parent in grads:
#                 # For gradient accumulation, we need to use regular addition
#                 # not the overloaded operator that creates new tracers
#                 if isinstance(grads[parent], Tracer) or isinstance(pg, Tracer):
#                     # If we have tracers, we need to trace through the accumulation
#                     grads[parent] = grads[parent] + pg
#                 else:
#                     # Regular numerical accumulation
#                     grads[parent] = grads[parent] + pg
#             else:
#                 grads[parent] = pg
    
    # return grads

def backward(node: Node, grad):
    topo = []
    visited = set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for p, _ in v.parents:
                build(p)
            topo.append(v)

    build(node)

    grads = {node: grad}

    for v in reversed(topo):
        g = grads[v]
        for parent, vjp in v.parents:
            pg = vjp(g)              # <-- MUST be raw torch tensor
            if pg is None:
                continue
            grads[parent] = grads.get(parent, 0) + pg

    return grads



def trace(f, args):
    flat, spec = flatten_pytree(args)

    leaves = []
    tracers = []

    for x in flat:
        n = Node(x, [])
        leaves.append(n)
        tracers.append(Tracer(x, n))

    traced_args = unflatten_pytree(tracers, spec)
    out = f(*traced_args)

    return out, leaves, spec


def unwrap(x): return getattr(x, 'value', x)

def value_and_grad(f):
    def wrapper(*args):
        out, leaves, spec = trace(f, args)

        if isinstance(out, (list, tuple)):
            out = out[0]

        init_grad = ONES_LIKE(unwrap(out))

        grads = backward(out.node, init_grad)

        flat_grads = []
        for leaf in leaves:
            g = grads.get(leaf, 0)
            flat_grads.append(g)

        grads_tree = unflatten_pytree(flat_grads, spec)

        return unwrap(out), grads_tree

    return wrapper

def grad(f):
    vg = value_and_grad(f)
    return lambda *a: vg(*a)[1]

