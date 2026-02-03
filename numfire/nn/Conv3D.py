from .base import Module
from .parameters import Variable
from .initializers import VarianceScaling, Constant
from ..src.DType import float32
from ..src.functions.primitive_arithmetic_and_basic_ops import matmul
from ..src.functions.convolution import convolution
from typing import Callable

def _ntuple(n, x):
    if isinstance(x, tuple):
        return x
    return (x,) * n

class Conv3D(Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: str | int | tuple[int, int, int] = "same",
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = True,
        weight_init: Callable | None = None,
        bias_init: Callable | None = None,
        name: str | None = None,
    ):
        super().__init__(name)

        self.out_c = out_channels
        self.kernel = _ntuple(3, kernel_size)
        self.stride = _ntuple(3, stride)
        self.dilation = _ntuple(3, dilation)
        self.padding = padding
        self.use_bias = bias

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
        # x: (N, Cin, D, H, W)
        N, Cin, D, H, W = x.shape

        if self.weight is None:
            w = self.weight_init(
                [self.out_c, Cin, self.kernel[0], self.kernel[1], self.kernel[2]],
                float32
            )
            self.weight = Variable(w)

        y = convolution(
            x=x,
            w=self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        if self.use_bias:
            if self.bias is None:
                b = self.bias_init([1, self.out_c, 1, 1, 1], float32)
                self.bias = Variable(b)
            y = y + self.bias

        return y
