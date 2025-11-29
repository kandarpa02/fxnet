from typing import Any
from .parameters import Variable, Parameter
from ..src.tree_util import register_tree_node


class Cell:
    def __init__(self, name: str = None):
        self.local_params: Parameter[Variable] = Parameter()
        self._cell_name = name  # optional cell name

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if isinstance(value, Variable):
            # attach hierarchical name
            prefix = getattr(self, "_cell_name", None)
            full_name = f"{prefix}.{name}" if prefix else name
            value.name = full_name
            self.local_params.append(value)

        elif isinstance(value, Cell):
            # propagate parent name
            value._cell_name = getattr(self, "_cell_name", None)
            if value._cell_name:
                value._cell_name += f".{name}"
            else:
                value._cell_name = name

    def parameters(self):
        for p in self.local_params:
            yield p
        for v in self.__dict__.values():
            if isinstance(v, Cell):
                yield from v.parameters()

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError


def _cell_flatten(cell: Cell):
    leaves = []
    meta = {
        "child_names": [],
        "param_names": [],
    }

    for name, value in cell.__dict__.items():
        if isinstance(value, Variable):
            meta["param_names"].append(name)
            leaves.append(value)
        elif isinstance(value, Cell):
            meta["child_names"].append(name)
            leaves.append(value)
        # Ignore other attributes (buffers, flags, cache, etc.)

    # flatten_fn must return (children_list, meta)
    return leaves, meta


def _cell_unflatten(children, meta):
    # Create an empty cell
    new = Cell()
    it = iter(children)

    # restore params
    for name in meta["param_names"]:
        setattr(new, name, next(it))

    # restore sub-cells
    for name in meta["child_names"]:
        setattr(new, name, next(it))

    return new


# Register Cell as pytree
register_tree_node(Cell, _cell_flatten, _cell_unflatten)
