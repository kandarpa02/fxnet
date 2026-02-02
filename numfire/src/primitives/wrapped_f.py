from .vjp_lib import *
from .utils_ops import *
from ..core import primitive
from ..DType import normalize_dtype, DType
DTypeLike = str|DType

def add(x, y): return primitive(lambda x, y: x + y, add_vjp)(x, y)

def sub(x, y): return primitive(lambda x, y: x - y, sub_vjp)(x, y)

def mul(x, y): return primitive(lambda x, y: x * y, mul_vjp)(x, y)

def div(x, y): return primitive(lambda x, y: x / y, div_vjp)(x, y)

def power(x, y): return primitive(lambda x, y: x ** y, pow_vjp)(x, y)

def maximum(x, y): return primitive(lambda x, y: torch.maximum(x, y), maximum_vjp)(x, y)

def minimum(x, y): return primitive(lambda x, y: torch.minimum(x, y), minimum_vjp)(x, y)

def neg(x): return primitive(lambda x: -x, neg_vjp)(x)

def exp(x): return primitive(lambda x: x.exp(), exp_vjp)(x)

def log(x): return primitive(lambda x: x.log(), log_vjp)(x)

def tanh(x): return primitive(lambda x: x.tanh(), tanh_vjp)(x)

def sigmoid(x): return primitive(lambda x: 1 / (1 + (-x).exp()), sigmoid_vjp)(x)

def relu(x): return primitive(lambda x: torch.relu(x), relu_vjp)(x)

def where(c, x, y): 
    return primitive(lambda c, x, y: torch.where(c, x, y), where_vjp)(c, x, y)

def sum(x, axis=None, keepdims=False):
    return primitive(
        lambda x: x.sum(dim=axis, keepdim=keepdims),
        lambda g, x: sum_vjp(g, x, axis, keepdims)
    )(x)

def mean(x, axis=None, keepdims=False):
    return primitive(
        lambda x: x.mean(dim=axis, keepdim=keepdims),
        lambda g, x: mean_vjp(g, x, axis, keepdims)
    )(x)

def reshape(x, shape):
    return primitive(
        lambda x: x.reshape(shape),
        lambda g, x: reshape_vjp(g, x, shape)
    )(x)

def transpose(x, axes):
    return primitive(
        lambda x: transpose_(x, axes),
        lambda g, x: transpose_vjp(g, x, axes=axes)
    )(x)

def matmul(x, y):
    return primitive(lambda x, y: x @ y, matmul_vjp)(x, y)

def equal(x, y):
    return primitive(lambda x, y: torch.eq(x, y), equal_vjp)(x, y)

def not_equal(x, y):
    return primitive(lambda x, y: torch.ne(x, y), not_equal_vjp)(x, y)

def greater(x, y):
    return primitive(lambda x, y: torch.gt(x, y), greater_vjp)(x, y)

def greater_equal(x, y):
    return primitive(lambda x, y: torch.ge(x, y), greater_equal_vjp)(x, y)

def less(x, y):
    return primitive(lambda x, y: torch.lt(x, y), less_vjp)(x, y)

def less_equal(x, y):
    return primitive(lambda x, y: torch.le(x, y), less_equal_vjp)(x, y)

def logical_not(x):
    return primitive(lambda x: torch.logical_not(x), logical_not_vjp)(x)

def logical_and(x, y):
    return primitive(lambda x, y: torch.logical_and(x, y), logical_and_vjp)(x)

def logical_or(x, y):
    return primitive(lambda x, y: torch.logical_or(x, y), logical_or_vjp)(x)

def logical_xor(x, y):
    return primitive(lambda x, y: torch.logical_xor(x, y), logical_xor_vjp)(x)

def logical_all(x, axis=None, keepdims=False):
    return primitive(
        lambda x: torch.all(x, dim=axis, keepdim=keepdims),
        lambda g, x: logical_all_vjp(g, x, axis, keepdims)
    )(x)

def logical_any(x, axis=None, keepdims=False):
    return primitive(
        lambda x: torch.any(x, dim=axis, keepdim=keepdims),
        lambda g, x: logical_any_vjp(g, x, axis, keepdims)
    )(x)

# Extra
def astype(x, dtype:DTypeLike):
    _dtype = normalize_dtype(dtype=dtype)
    return primitive(
        lambda buf: buf.to(_dtype),
        lambda g, buf: astype_vjp(g, buf)
    )(x)
