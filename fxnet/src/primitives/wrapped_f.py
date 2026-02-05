from .vjp_lib import *
from .utils_ops import *
from ..core import primitive
from ..DType import normalize_dtype, DType
DTypeLike = str|DType

import torch as T

def add(x, y): return primitive(lambda x, y: T.add(x, y), add_vjp)(x, y)

def sub(x, y): return primitive(lambda x, y: T.sub(x, y), sub_vjp)(x, y)

def mul(x, y): return primitive(lambda x, y: T.mul(x, y), mul_vjp)(x, y)

def div(x, y): return primitive(lambda x, y: T.div(x, y), div_vjp)(x, y)

def power(x, y): return primitive(lambda x, y: T.pow(x, y), pow_vjp)(x, y)


def maximum(x, y): return primitive(lambda x, y: T.maximum(x, y), maximum_vjp)(x, y)

def minimum(x, y): return primitive(lambda x, y: T.minimum(x, y), minimum_vjp)(x, y)


def neg(x): return primitive(lambda x: T.neg(x), neg_vjp)(x)

def exp(x): return primitive(lambda x: T.exp(x), exp_vjp)(x)

def log(x): return primitive(lambda x: T.log(x), log_vjp)(x)

def tanh(x): return primitive(lambda x: T.tanh(x), tanh_vjp)(x)

def sigmoid(x): return primitive(lambda x: T.sigmoid(x), sigmoid_vjp)(x)

def relu(x): return primitive(lambda x: T.relu(x), relu_vjp)(x)


def where(c, x, y):
    return primitive(lambda c, x, y: T.where(c, x, y), where_vjp)(c, x, y)


def sum(x, axis=None, keepdims=False):
    return primitive(
        lambda x: T.sum(x, dim=axis, keepdim=keepdims),
        lambda g, x: sum_vjp(g, x, axis, keepdims)
    )(x)

def mean(x, axis=None, keepdims=False):
    return primitive(
        lambda x: T.mean(x, dim=axis, keepdim=keepdims),
        lambda g, x: mean_vjp(g, x, axis, keepdims)
    )(x)

def reshape(x, shape):
    return primitive(
        lambda x: T.reshape(x, shape),
        lambda g, x: reshape_vjp(g, x, shape)
    )(x)

def transpose(x, axes):
    return primitive(
        lambda x: T.permute(x, axes),
        lambda g, x: transpose_vjp(g, x, axes=axes)
    )(x)

def matmul(x, y):
    return primitive(lambda x, y: T.matmul(x, y), matmul_vjp)(x, y)


# comparisons

def equal(x, y): return primitive(lambda x, y: T.eq(x, y), equal_vjp)(x, y)
def not_equal(x, y): return primitive(lambda x, y: T.ne(x, y), not_equal_vjp)(x, y)
def greater(x, y): return primitive(lambda x, y: T.gt(x, y), greater_vjp)(x, y)
def greater_equal(x, y): return primitive(lambda x, y: T.ge(x, y), greater_equal_vjp)(x, y)
def less(x, y): return primitive(lambda x, y: T.lt(x, y), less_vjp)(x, y)
def less_equal(x, y): return primitive(lambda x, y: T.le(x, y), less_equal_vjp)(x, y)


# logical

def logical_not(x):
    return primitive(lambda x: T.logical_not(x), logical_not_vjp)(x)

def logical_and(x, y):
    return primitive(lambda x, y: T.logical_and(x, y), logical_and_vjp)(x, y)

def logical_or(x, y):
    return primitive(lambda x, y: T.logical_or(x, y), logical_or_vjp)(x, y)

def logical_xor(x, y):
    return primitive(lambda x, y: T.logical_xor(x, y), logical_xor_vjp)(x, y)

def logical_all(x, axis=None, keepdims=False):
    return primitive(
        lambda x: T.all(x, dim=axis, keepdim=keepdims),
        lambda g, x: logical_all_vjp(g, x, axis, keepdims)
    )(x)

def logical_any(x, axis=None, keepdims=False):
    return primitive(
        lambda x: T.any(x, dim=axis, keepdim=keepdims),
        lambda g, x: logical_any_vjp(g, x, axis, keepdims)
    )(x)


# Extra

def astype(x, dtype: DTypeLike):
    _dtype = normalize_dtype(dtype=dtype)
    return primitive(
        lambda buf: T.Tensor.to(buf, _dtype),
        lambda g, buf: astype_vjp(g, buf)
    )(x)
