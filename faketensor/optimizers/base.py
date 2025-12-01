from ..nn.base import Cell
from ..nn.parameters import Variable, NDarray
from typing import Dict, Any


class Optimizer:
    def __init__(self, model: Cell):
        self.model = model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.get_state()})"

    def get_state(self) -> Dict[str, Any]:
        """Return all optimizer hyperparameters & buffers except model."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if k != "model"
        }
    
    def load_state(self, state: Dict[str, Any]):
        """
        Load state safely:
        - only update keys that exist in the optimizer
        - ignore extra keys (forward-compatible)
        """
        for k, v in state.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def update_rule(self, **kwargs):
        raise NotImplementedError

    def update(self, grads):
        new_params = self.update_rule(grads=grads)
        self.model.parameters_upload(new_params)
