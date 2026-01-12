from .base import Optimizer
from ..nn.base import Cell

class GradientDescent(Optimizer):
    """Gradient Descent optimizer.

    This optimizer implements the standard gradient descent update:

    ```
    variable = variable - learning_rate * gradient
    ```

    No momentum or weight decay is applied. This is equivalent to
    `tf.keras.optimizers.SGD` with `momentum=0.0`.

    Args:
        model: A `Cell` instance whose trainable parameters will be updated.
        lr: Float. Learning rate. Defaults to `0.1`.

    Returns:
        A list of updated parameters, in the same order as returned by
        `model.trainable_parameters()`.

    Notes:
        The returned parameters must be uploaded back to the model using
        `model.parameters_upload(updated_params)`.
    """
    def __init__(self, model:Cell, lr=0.1):
        super().__init__(model)
        self.lr = lr

    def update_rule(self, grads):
        new_params = [
            p - self.lr * g
            for p, g in zip(self.model.trainable_parameters(), grads)
        ]
        return new_params


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with optional momentum,
    Nesterov acceleration, and weight decay.

    This optimizer implements the following update rules:

    Weight decay (L2 regularization):

    ```
    gradient = gradient + weight_decay * parameter
    ```

    Momentum:

    ```
    velocity = momentum * velocity + gradient
    ```

    Parameter update:

    - If `nesterov == False`:

      ```
      parameter = parameter - learning_rate * velocity
      ```

    - If `nesterov == True`:

      ```
      parameter = parameter - learning_rate * (gradient + momentum * velocity)
      ```

    Behavior matches the semantics of `tf.keras.optimizers.SGD` with
    `momentum` and `nesterov` options.

    Args:
        model: A `Cell` instance whose trainable parameters will be updated.
        lr: Float. Learning rate.
        momentum: Float in `[0, 1)`. Defaults to `0.0`. Set to a positive
            value to enable momentum.
        nesterov: Boolean. If `True`, uses Nesterov accelerated gradient.
            Ignored if `momentum == 0.0`.
        weight_decay: Float. L2 regularization factor. Defaults to `0.0`.

    Returns:
        A list of updated parameters, matching the order of
        `model.trainable_parameters()`.

    Attributes:
        velocity: Dictionary mapping parameter IDs to momentum buffers.

    Notes:
        - Momentum buffers are created lazily during the first update.
        - Weight decay is applied to the gradients before momentum.
        - All updates are functional; the returned parameter list must be
          passed to `model.parameters_upload()`.

    """
    
    def __init__(
        self,
        model: Cell,
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ):
        super().__init__(model)

        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.velocity = {}

    def update_rule(self, grads):
        params = self.model.trainable_parameters()
        new_params = []

        for p, g in zip(params, grads):

            # ---------- Weight Decay ----------
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p

            # ---------- Momentum ----------
            if self.momentum != 0.0:
                pid = id(p)

                # initialize buffer if missing
                if pid not in self.velocity:
                    self.velocity[pid] = g * 0.0   # same shape, zeros

                v = self.velocity[pid]
                v_new = self.momentum * v + g
                self.velocity[pid] = v_new

                # ---------- Nesterov ----------
                if self.nesterov:
                    # θ ← θ − lr * (g + μ v_new)
                    update = g + self.momentum * v_new
                else:
                    # θ ← θ − lr * v_new
                    update = v_new

            else:
                # no momentum at all → normal SGD
                update = g

            # ---------- Apply update ----------
            new_p = p - self.lr * update
            new_params.append(new_p)

        return new_params
