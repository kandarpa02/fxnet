import dataclasses
import typing
import numpy as np


# ---------- Tracing objects ----------

class Tracer:
    def __init__(self, value, node=None):
        self.value = value
        self.node = node

    def __repr__(self):
        return f"Tracer({self.value})"


@dataclasses.dataclass
class Node:
    parents: typing.Sequence["Tracer"]
    bwd: typing.Callable  # bwd(g) -> grads for parents


# ---------- Primitive wrapper ----------

def primitive(fwd, vjp):
    def wrapped(*args):
        # unwrap values
        vals = [a.value if isinstance(a, Tracer) else a for a in args]
        parents = [a for a in args if isinstance(a, Tracer)]

        out = fwd(*vals)

        # no tracer inputs -> pure forward
        if not parents:
            return out

        # build VJP node
        def bwd(g):
            return vjp(g, *vals)

        node = Node(parents, bwd)
        return Tracer(out, node)

    return wrapped


# ---------- Engine (super dumb) ----------

def backward(root: Tracer):
    grads = {root: np.ones_like(root.value)}
    stack = [root]

    while stack:
        t = stack.pop()
        node = t.node
        if node is None:
            continue

        gout = grads[t]
        parent_grads = node.bwd(gout)

        for parent, g in zip(node.parents, parent_grads):
            if parent in grads:
                grads[parent] = grads[parent] + g
            else:
                grads[parent] = g
                stack.append(parent)

    return grads


# ---------- grad with argnums ----------

def grad(f, argnums=0):
    # ---- normalize argnums ----
    if isinstance(argnums, int):
        argnums = (argnums,)
    else:
        argnums = tuple(argnums)

    # remove duplicates but preserve order
    seen = set()
    argnums = tuple(i for i in argnums if not (i in seen or seen.add(i)))

    def grad_f(*args):
        tracers = list(args)

        # ---- inject tracers ----
        for i in argnums:
            tracers[i] = Tracer(args[i])

        out = f(*tracers)

        if not isinstance(out, Tracer):
            raise ValueError("grad requires the function to return a scalar.")

        grads = backward(out)

        # ---- collect grads in same order as argnums ----
        results = []
        for i in argnums:
            t = tracers[i]
            g = grads.get(t, np.zeros_like(t.value))
            results.append(g)

        return results[0] if len(results) == 1 else tuple(results)

    return grad_f



mul = primitive(
    lambda x, y: np.multiply(x, y),
    lambda g, x, y: (g * y, g * x),
)

add = primitive(
    lambda x, y: np.add(x, y),
    lambda g, x, y: (g, g),
)

sin = primitive(
    lambda x: np.sin(x),
    lambda g, x: (g * np.cos(x),),
)

def square(x):
    return mul(x, x)

def f(x, y):
    return add(square(x), mul(x, y))

d = grad(f, argnums=[0, 1])

print(d(3.0, 4.0))

print(type(f(3., 4.)))