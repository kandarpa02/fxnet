from .core import Node
from .tracer import Tracer
from ..tree_util import flatten_pytree, unflatten_pytree
from .basic_functions.non_grad_utils import ONES_LIKE

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
    flat_args, arg_def = flatten_pytree(args)

    leaves = []
    tracers = []
    for arg in flat_args:
        leaf = Node(arg, [])
        leaves.append(leaf)
        tracers.append(Tracer(arg, leaf))
    
    traced_args = unflatten_pytree(tracers, arg_def)
    out = f(*traced_args)
    return out, leaves

# ADD PYTREE TO IT
def return_grads_and_others_(f):
    def df(*args):
        # Trace the function to build computational graph
        # flat_args, arg_def = flatten_pytree(args)
        out, leaves = trace(f, args)
        
        # Initialize gradient
        init_grad = ONES_LIKE(out.value)
        
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

