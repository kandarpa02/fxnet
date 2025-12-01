from .base import Optimizer
from ..nn.base import Cell

class GradientDescent(Optimizer):
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
