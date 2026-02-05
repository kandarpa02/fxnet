from .vjp_lib import liner_vjp, conv_general_vjp
from . import utils_ops as U
from ..core import primitive
from .wrapped_f import matmul, add

def linear(x, w, b=None):
    def func(x, w, b):
        y = matmul(x, w)
        if b is not None:
            return add(y, b)
        return y

    lin_f = primitive(func, lambda g: liner_vjp(x, w, b))
    return lin_f(x, w, b)

# 'stride', 'dilation', 'padding', 'dims', 'kernel_shape'

# primitive(pad)
# primitive(unfold)
# primitive(fold)
# primitive(einsum)
# primitive(reshape/view)
# primitive(transpose/permute)
# primitive(slice)


def conv_general(x, w, stride=1, padding="valid", dilation=1):
    out, pads = U.convolution_f(x, w, stride, padding, dilation)

    def vjp(g, x, w):
        return conv_general_vjp(g, x, w, stride, padding, dilation, pads)

    return primitive(lambda x, w: out, vjp)(x, w)
