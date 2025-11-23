from dataclasses import is_dataclass, fields
from typing import Any, Callable, List, Tuple, Dict, Type

# Registry for custom container types
_PYTREE_REGISTRY: Dict[Type, Callable] = {}


def register_tree_node(typ, flatten_fn, unflatten_fn):
    _PYTREE_REGISTRY[typ] = (flatten_fn, unflatten_fn)


class TreeDef:
    """Structural definition of the pytree."""
    def __init__(self, typ, children):
        self.typ = typ
        self.children = children

    def __repr__(self):
        return f"TreeDef({self.typ}, {self.children})"


def flatten_pytree(x: Any) -> Tuple[List[Any], TreeDef]:
    """Flatten a Python object into (leaves, treedef)."""

    for typ, (flatten_fn, _) in _PYTREE_REGISTRY.items():
        if isinstance(x, typ):
            children, meta = flatten_fn(x)
            flat_children = []
            child_treedefs = []
            for c in children:
                leaves, t = flatten_pytree(c)
                flat_children.extend(leaves)
                child_treedefs.append(t)
            return flat_children, TreeDef((typ, meta), child_treedefs)

    if isinstance(x, dict):
        flat_children = []
        child_treedefs = []
        keys = tuple(x.keys())
        for k in keys:
            leaves, t = flatten_pytree(x[k])
            flat_children.extend(leaves)
            child_treedefs.append((k, t))
        return flat_children, TreeDef(("dict", keys), child_treedefs)

    if isinstance(x, (list, tuple)):
        flat_children = []
        child_treedefs = []
        for elem in x:
            leaves, t = flatten_pytree(elem)
            flat_children.extend(leaves)
            child_treedefs.append(t)
        return flat_children, TreeDef(x.__class__, child_treedefs)

    if is_dataclass(x):
        flat_children = []
        child_treedefs = []
        fs = fields(x)
        for f in fs:
            v = getattr(x, f.name)
            leaves, t = flatten_pytree(v)
            flat_children.extend(leaves)
            child_treedefs.append((f.name, t))
        return flat_children, TreeDef(("dataclass", x.__class__), child_treedefs)

    # Base case — leaf
    return [x], TreeDef("leaf", None)


def unflatten_pytree(leaves: List[Any], treedef: TreeDef) -> Any:
    it = iter(leaves)
    return _unflatten(it, treedef)


def _unflatten(it, treedef: TreeDef):
    typ = treedef.typ

    # Leaf
    if typ == "leaf":
        return next(it)

    # dict
    if isinstance(typ, tuple) and typ[0] == "dict":
        keys = typ[1]
        out = {}
        for (k, child_def) in treedef.children:
            out[k] = _unflatten(it, child_def)
        return out

    # tuple/list
    if typ in (list, tuple):
        elems = [_unflatten(it, child_def) for child_def in treedef.children]
        return typ(elems)

    # dataclass
    if isinstance(typ, tuple) and typ[0] == "dataclass":
        cls = typ[1]
        kwargs = {}
        for (name, child_def) in treedef.children:
            kwargs[name] = _unflatten(it, child_def)
        return cls(**kwargs)

    # Custom registered node
    for base_typ, (_, unflatten_fn) in _PYTREE_REGISTRY.items():
        if isinstance(typ, tuple) and typ[0] is base_typ:
            meta = typ[1]
            children = [_unflatten(it, c) for c in treedef.children]
            return unflatten_fn(children, meta)

    raise TypeError(f"Unsupported treedef: {treedef}")
