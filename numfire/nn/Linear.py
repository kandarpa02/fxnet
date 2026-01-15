from .base import Cell
from .parameters import Variable
from .initializers import VarianceScaling, Constant
from ..src.DType import float32
from ..src.functions.primitive_arithmetic_and_basic_ops import matmul

class Linear(Cell):
    """Applies a fully connected linear transformation to the input.

    This layer computes the transformation:

        ```
        y = xW + b
        ```
        
    where:
    - `W` is a learnable weight matrix of shape `(in_feat, out_feat)`.
    - `b` is an optional learnable bias vector of shape `(out_feat,)`.

    By default:
    - Weights are initialized using a **Xavier-Normal initializer**
      (`VarianceScaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")`),
      suitable for ReLU activations.
    - Biases are initialized to zero.

    Parameters:
        in_feat: Number of input features (size of the last dimension of input).
        out_feat: Number of output features.
        bias: Whether to include a bias term. Defaults to True.
        weight_init: Optional custom initializer for weights.  
            If None, uses Xavier initialization.
        bias_init: Optional custom initializer for bias.  
            If None, uses a constant zero initializer.
        dtype: Mathino data type for parameters. Defaults to `mt.float32`.
        name: Optional string name for variable scoping.
    """

    def __init__(self, in_features, out_features, bias=True, weight_init=None, bias_init=None, name: str | None = None):
        super().__init__(name)
        self.if_bias = bias
        w_init = VarianceScaling(scale=1.0, mode='fan_avg')([in_features, out_features], dtype=float32) if weight_init is None else weight_init
        b_init = Constant(0.0)(out_features, float32) if bias_init is None else bias_init

        self.weight = Variable(w_init)
        if bias:
            self.bias = Variable(b_init)
    
    def call(self, x):
        y = matmul(x, self.weight)
        if self.if_bias:
            return y + self.bias
        return y