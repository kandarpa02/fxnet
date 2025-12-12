from typing import Any, Generator
from .parameters import Variable, Parameter
from ..src.tree_util import register_tree_node


class Cell:
    """Base class for all neural network components in FakeTensor.

    A `Cell` is the fundamental building block of the FakeTensor neural
    network API. It is conceptually similar to `tf.keras.layers.Layer`.
    A `Cell`:

    - manages hierarchical subcells (modules)
    - registers trainable `Variable` parameters
    - maintains unique fully-qualified parameter names
    - supports functional parameter iteration
    - integrates with FakeTensor's pytree system

    Subclasses must override the `call()` method to define computation.

    Args:
        name: Optional string. The base name for this cell. If not provided,
            the class name is used.

    Attributes:
        _cell_name: The fully qualified name of this cell in the module
            hierarchy.
        local_params: A `Parameter` container holding trainable Variables
            defined directly in this cell (not including child cells).

    Notes:
        - Assigning a `Variable` or another `Cell` to an attribute automatically
          updates its hierarchical name (e.g., `"MLP.layer1.weight"`).
        - Calling a `Cell` (via `__call__`) forwards directly to `call()`.
        - `parameters()` returns all parameters recursively.
        - `trainable_parameters()` filters only parameters with `train=True`.

    Example:
        ```python
        class Linear(Cell):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.w = Variable(randn(in_dim, out_dim), name="w")
                self.b = Variable(zeros(out_dim), name="b")

            def call(self, x):
                return x @ self.w + self.b

        model = Linear(4, 3)
        out = model(x)  # invokes model.call(x)
        ```
    """

    def __init__(self, name: str|None = None):   #type:ignore
        super().__setattr__("_cell_name", name if name is not None else self.__class__.__name__)
        super().__setattr__("local_params", Parameter())

    def __repr__(self):
        body = f"{self.__class__.__name__}("
        for k, mod in self.__dict__.items():
            if isinstance(mod, Cell):
                body += f"\n   ({k}): {repr(mod)}"
            
            else:
                mod.__repr__()
        return body + "\n)"


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
    def parameters(self, return_tuple=False)->tuple[Parameter]|Generator:
        """Returns all parameters (trainable + non-trainable) in this cell.

        This includes parameters defined in:
        - this cell (`local_params`)
        - all descendant subcells (recursively)

        Returns:
            A generator yielding `Variable` objects.
        """

        def gather():
            for p in self.local_params:   #type:ignore
                yield p 

            for v in self.__dict__.values():
                if isinstance(v, Cell):
                    yield from v.parameters()

        return tuple(gather()) if return_tuple else gather()


    def trainable_parameters(self, return_tuple=False)->tuple[Parameter]|Generator:
        """Returns all trainable parameters in this cell.

        A parameter is considered trainable if `param.train == True`.

        Returns:
            A generator yielding trainable `Variable` objects.
        """
        def gather():
            for p in self.local_params:  #type:ignore
                if p.train:
                    yield p
                else:
                    pass

            for v in self.__dict__.values():
                if isinstance(v, Cell):
                    yield from v.trainable_parameters()

        return tuple(gather()) if return_tuple else gather()

    def parameters_upload(self, new_params):
        """Replaces trainable parameters with an updated list.

        This method is called by optimizers after computing new parameter
        values. Parameters are updated in the order returned by
        `trainable_parameters()`.

        Args:
            new_params: A list of updated `Variable` objects or NDArrays
                containing new data.

        Notes:
            This method performs an in-place data copy into each parameter’s
            internal array buffer (`old.np[...] = new.np`).
        """

        for old, new in zip(self.trainable_parameters(), new_params):
            old.np[...] = new.np

    def apply(self, params, *args, **kwargs):
        self.parameters_upload(params)
        return self(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        """Defines the forward pass logic.

        Subclasses must override this method. It is invoked when calling the
        cell directly:

        ```
        output = cell(inputs)
        ```

        Args:
            *args: Positional arguments for the forward pass.
            **kwargs: Keyword arguments for the forward pass.

        Returns:
            The output of the forward computation.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError


def flatten(cell: Cell):
    """PyTree flatten function for Cell.

    Extracts Variables and subcells as leaves, and stores metadata
    describing their names.

    Returns:
        leaves: A list of Variables and child Cells.
        meta: A dictionary describing param and child attribute names.
    """

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
    """PyTree unflatten function for Cell.

    Reconstructs a Cell from its flattened children and metadata.

    Args:
        children: List of leaves (Variables and Cells).
        meta: Metadata dictionary produced during flattening.

    Returns:
        A reconstructed `Cell` with restored structure.
    """
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
