from .vjp_lib import liner_vjp, conv_general_vjp
from . import utils_ops as U
from ..core import primitive
from .wrapped_f import matmul, add

def linear(x, w, b=None):
    def func(x, w, b):
        y = matmul(x, w)
        if b:
            return add(y, b)
        return y

    lin_f = primitive(func, lambda g: liner_vjp(x, w, b))
    return lin_f(x, w, b)

