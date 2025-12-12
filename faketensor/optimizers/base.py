from ..nn.base import Cell
from ..nn.parameters import Variable, NDarray
from typing import Dict, Any


class Optimizer:
    """Base class for all optimizers in FakeTensor.

    This class defines the common interface shared by all optimizers.
    Subclasses must override `update_rule()` to implement the actual
    optimization algorithm. The `update()` method applies the computed
    updates to the model's trainable parameters.

    Args:
        model: A `Cell` instance. The model whose trainable parameters
            will be updated by this optimizer.

    Methods:
        get_state():
            Returns a dictionary containing all optimizer hyperparameters,
            internal buffers (e.g., momentum vectors), and other persistent
            variables. The model reference is excluded.

        load_state(state):
            Loads a previously saved optimizer state. Unknown or extra keys
            in the input dictionary are safely ignored.

        update_rule(**kwargs):
            Computes and returns the updated parameters. Must be implemented
            by subclasses. Should not modify parameters in-place.

        update(grads):
            Computes the new parameters using `update_rule()` and uploads
            them into the model using `model.parameters_upload()`.

    Notes:
        - Optimizers in FakeTensor operate in a functional style:
          `update_rule()` returns a list of new parameter tensors, and the
          model is updated externally.
        - Internal states (e.g., momentum buffers) are stored inside the
          optimizer instance.
        - Subclasses should follow the same pattern as TensorFlow/Keras
          optimizers, but adapted to the FakeTensor design.
    """

    def __init__(self, model: Cell):
        self.model = model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.get_state()})"

    def get_state(self) -> Dict[str, Any]:
        """Returns the optimizer's hyperparameters and internal state.

        Excludes the model reference. Useful for checkpointing and restoring
        training sessions.

        Returns:
            A dictionary mapping state names to values.
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if k != "model"
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Loads a saved optimizer state safely.

        Only keys present in the current optimizer instance are restored.
        Extra or unknown keys are ignored, allowing forward-compatible
        checkpoints.

        Args:
            state: A dictionary returned by `get_state()`.
        """
        for k, v in state.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def update_rule(self, **kwargs):
        """Computes new parameter values.

        Subclasses must implement this method.

        Args:
            **kwargs: Optimizer-specific keyword arguments, usually including
                gradients.

        Returns:
            A list of new parameter tensors.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    def update(self, grads):
        """Applies a single optimization step.

        This method:
        1. Computes updated parameters via `update_rule(grads=grads)`.
        2. Uploads them to the model using `model.parameters_upload()`.

        Args:
            grads: A list of gradients corresponding to the model’s
                trainable parameters.
        """
        new_params = self.update_rule(grads=grads)
        self.model.parameters_upload(new_params)


    def update_and_get_params(self, grads):
        """Applies a single optimization step.

        This method:
        1. Computes updated parameters via `update_rule(grads=grads)`.
        2. Returns the params for functional usage.

        Args:
            grads: A list of gradients corresponding to the model’s
                trainable parameters.
        """
        return self.update_rule(grads=grads)