from typing import Sequence
from .base import Module, Layer
from ..src.tensor_value import Variable
from .initializers import VarianceScaling, Constant
from ..src.DType import float32
from ..src.primitives.wrapped_f import matmul

class Dense(Module):
    def __init__(
        self,
        out_features,
        bias=True,
        weight_init=None,
        bias_init=None,
        name: str | None = None
    ):
        super().__init__(name)

        self.units = out_features
        self.if_bias = bias

        self.weight_init = (
            weight_init
            if weight_init is not None
            else VarianceScaling(scale=1.0, mode="fan_avg")
        )

        self.bias_init = (
            bias_init
            if bias_init is not None
            else Constant(0.0)
        )

        self.weight: Variable | None = None
        self.bias: Variable | None = None

    def call(self, x):
        if self.weight is None:
            w = self.weight_init([x.shape[-1], self.units], float32)
            self.weight = Variable(w)

        y = matmul(x, self.weight)

        if self.if_bias:
            if self.bias is None:
                b = self.bias_init([self.units], float32)
                self.bias = Variable(b)
            y = y + self.bias

        return y
    
    