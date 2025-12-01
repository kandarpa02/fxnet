from typing import Any
from .parameters import Variable, Parameter
from ..src.tree_util import register_tree_node


class Cell:
    def __init__(self, name: str = None):   #type:ignore
        super().__setattr__("_cell_name", name if name is not None else self.__class__.__name__)
        super().__setattr__("local_params", Parameter())

    def _full_child_prefix(self, child_name):
        if self._cell_name is None: #type:ignore
            return child_name
        return f"{self._cell_name}.{child_name}" #type:ignore

    def _update_param_names(self):
        """Rename parameters after parent attaches this Cell."""
        for name, v in self.__dict__.items():
            if isinstance(v, Variable):
                v.name = f"{self._cell_name}.{name}" #type:ignore

            elif isinstance(v, Cell):
                v._cell_name = f"{self._cell_name}.{name}" #type:ignore
                v._update_param_names()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        # --------------------------
        # Case: assigning submodule
        # --------------------------
        if isinstance(value, Cell):
            # assign correct hierarchical name
            if self._cell_name:  #type:ignore
                value._cell_name = f"{self._cell_name}.{name}" #type:ignore
            else:
                value._cell_name = name

            # now rename all its parameters recursively!
            value._update_param_names()

        # --------------------------
        # Case: assigning Variable
        # --------------------------
        elif isinstance(value, Variable):
            prefix = self._cell_name  #type:ignore
            if prefix:
                value.name = f"{prefix}.{name}"
            else:
                value.name = name

            self.local_params.append(value)  #type:ignore

    # Parameter recursion
    def parameters(self):
        for p in self.local_params:   #type:ignore
            yield p

        for v in self.__dict__.values():
            if isinstance(v, Cell):
                yield from v.parameters()

    def trainable_parameters(self):
        for p in self.local_params:  #type:ignore
            if p.train:
                yield p
            else:
                pass

        for v in self.__dict__.values():
            if isinstance(v, Cell):
                yield from v.trainable_parameters()

    def parameters_upload(self, new_params):
        for old, new in zip(self.trainable_parameters(), new_params):
            old.np[...] = new.np

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError


def flatten(cell: Cell):
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


def unflatten(children, meta):
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
register_tree_node(Cell, flatten, unflatten)
