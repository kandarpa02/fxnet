from ..core import primitive
from .utils import unbroadcast
import torch
from collections.abc import Sequence


__all__ = [
    # basic arithmetic
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "pow",
    
    # unary math
    "exp",
    "log",
    "sin",
    "cos",
    "tanh",
    "sign",
    "abs",

    # reductions 
    "sum",
    "mean",
    "prod",
    "max",
    "min",

    # linear algebra
    "matmul",

    # shape ops
    "reshape",
    "permute",
    "squeeze",
    "unsqueeze",
    "broadcast_to",

    # comparison
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "equal",
    "not_equal",

    # logical
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",

    # selection
    "where",
    "maximum",
    "minimum",
    "clamp",

    # tensor combining
    "cat",
    "stack",
]


# ---------------- add ----------------

@primitive
def add(x, y):
    return torch.add(x, y)

add.defvjp(
    lambda x, y: (torch.add(x, y), [x, y]),
    lambda g, res: (
        unbroadcast(res[0], g),
        unbroadcast(res[1], g),
    )
)



# ---------------- sub ----------------

@primitive
def sub(x, y):
    return torch.sub(x, y)

sub.defvjp(
    lambda x, y: (torch.sub(x, y), [x, y]),
    lambda g, res: (
        unbroadcast(res[0], g),
        unbroadcast(res[1], -g),
    )
)


# ---------------- mul ----------------

@primitive
def mul(x, y):
    return torch.mul(x, y)

mul.defvjp(
    lambda x, y: (torch.mul(x, y), [x, y]),
    lambda g, res: (
        unbroadcast(res[0], g * res[1]),
        unbroadcast(res[1], g * res[0]),
    )
)


# ---------------- div ----------------

@primitive
def div(x, y):
    return torch.div(x, y)

div.defvjp(
    lambda x, y: (torch.div(x, y), [x, y]),
    lambda g, res: (
        unbroadcast(res[0], g / res[1]),
        unbroadcast(res[1], -g * res[0] / (res[1] ** 2)),
    )
)


# ---------------- neg ----------------

@primitive
def neg(x):
    return torch.neg(x)

neg.defvjp(
    lambda x: (torch.neg(x), [x]),
    lambda g, res: (-g,)
)


# ---------------- pow ----------------

@primitive
def pow(x, y):
    return torch.pow(x, y)

def pow_f(x, y):
    out = torch.pow(x, y)
    logx = torch.log(x)
    return out, (x, y, out)

def pow_vjp(g, res):
    x, y, z = res
    return (
        unbroadcast(x, g*y*z/x),
        unbroadcast(y, g*x.log()*z)
    )

pow.defvjp(
    pow_f,
    pow_vjp
)

# ---------------- exp ----------------

@primitive
def exp(x):
    return torch.exp(x)

def exp_f(x):
    out = torch.exp(x)
    return out, [out]

exp.defvjp(
    lambda x: exp_f(x),
    lambda g, res: (g * res[0],)
)


# ---------------- log ----------------

@primitive
def log(x):
    return torch.log(x)

log.defvjp(
    lambda x: (torch.log(x), [x]),
    lambda g, res: (g / res[0],)
)


# ---------------- sin ----------------

@primitive
def sin(x):
    return torch.sin(x)

sin.defvjp(
    lambda x: (torch.sin(x), [x]),
    lambda g, res: (g * res[0].cos(),)
)


# ---------------- cos ----------------

@primitive
def cos(x):
    return torch.cos(x)

cos.defvjp(
    lambda x: (torch.cos(x), [x]),
    lambda g, res: (-g * res[0].sin(),)
)


# ---------------- tanh ----------------

@primitive
def tanh(x):
    return torch.tanh(x)

tanh.defvjp(
    lambda x: (torch.tanh(x), [torch.tanh(x)]),
    lambda g, res: (g * (1 - res[0] ** 2),)
)


# ---------------- sum ----------------


def sum(x, axis=None, keepdims=False):
    @primitive
    def _sum(x):
        return torch.sum(x, dim=axis, keepdim=keepdims)

    _sum.defvjp(
        lambda x: (torch.sum(x, dim=axis, keepdim=keepdims), [x]),
        lambda g, res: (torch.ones_like(res[0]) * g,)
    )
    return _sum(x)

# ---------------- mean ----------------
def mean(x, axis=None, keepdims=False):
    @primitive
    def _mean(x):
        return torch.mean(x, dim=axis, keepdim=keepdims)

    mean.defvjp(
        lambda x: (torch.mean(x, dim=axis, keepdim=keepdims), [x]),
        lambda g, res: (torch.ones_like(res[0]) * g / res[0].numel(),)
    )
    return _mean(x)

def prod(x, axis=None, keepdims=False):
    @primitive
    def _prod(x):
        return torch.prod(x, dim=axis, keepdim=keepdims)

    _prod.defvjp(
        lambda x: (
            torch.prod(x, dim=axis, keepdim=keepdims),
            [x, torch.prod(x, dim=axis, keepdim=keepdims)],
        ),
        lambda g, res: (
            g * res[1] / res[0],
        )
    )

    return _prod(x)

def max(x, axis=None, keepdims=False):
    @primitive
    def _max(x):
        return torch.max(x, dim=axis, keepdim=keepdims).values

    _max.defvjp(
        lambda x: (
            torch.max(x, dim=axis, keepdim=keepdims).values,
            [x, torch.max(x, dim=axis, keepdim=keepdims).values],
        ),
        lambda g, res: (
            (res[0] == res[1]) * g,
        )
    )

    return _max(x)

def min(x, axis=None, keepdims=False):
    @primitive
    def _min(x):
        return torch.min(x, dim=axis, keepdim=keepdims).values

    _min.defvjp(
        lambda x: (
            torch.min(x, dim=axis, keepdim=keepdims).values,
            [x, torch.min(x, dim=axis, keepdim=keepdims).values],
        ),
        lambda g, res: (
            (res[0] == res[1]) * g,
        )
    )

    return _min(x)


# ---------------- matmul ----------------

@primitive
def matmul(x, y):
    return torch.matmul(x, y)

matmul.defvjp(
    lambda x, y: (torch.matmul(x, y), [x, y]),
    lambda g, res: (
        unbroadcast(res[0], g @ res[1].T),
        unbroadcast(res[1], res[0].T @ g),
    )
)

def _getitem(x, idx):
    @primitive
    def getitem(x):
        return torch.Tensor.__getitem__(x, idx)

    def _scatter_like(x, idx, g):
        out = torch.zeros_like(x)
        out[idx] = out[idx] + g
        return out

    getitem.defvjp(
        # fwd: pure torch
        lambda x: (
            torch.Tensor.__getitem__(x, idx),
            [x],  
        ),

        # bwd: Texor world only
        lambda g, res: (
            _scatter_like(res[0], res[1], g),
        )
    )
    return getitem(x)


def reshape(x, shape:Sequence[int]):
    @primitive
    def _reshape(x):
        return torch.reshape(x, shape)
    
    _reshape.defvjp(
        lambda x: (torch.reshape(x, shape), [x]),
        lambda g, res: (g.reshape(*res[0].shape),)
    )
    return _reshape(x)


def permute(x, axes):
    @primitive
    def _permute(x):
        return torch.permute(x, dims=axes)

    def _invert_permutation(axes):
        inv = [0] * len(axes)
        for i, d in enumerate(axes):
            inv[d] = i
        return tuple(inv)

    _permute.defvjp(
        lambda x: (torch.permute(x, axes), []),

        lambda g, res: (
            g.permute(*_invert_permutation(axes)),
            *([None] * len(axes)),
        )
    )
    return _permute(x)

def greater(x, y):
    @primitive
    def _greater(x, y):
        return torch.gt(x, y)

    _greater.defvjp(
        lambda x, y: (torch.gt(x, y), []),
        lambda g, res: (None, None)
    )

    return _greater(x, y)


def greater_equal(x, y):
    @primitive
    def _ge(x, y):
        return torch.ge(x, y)

    _ge.defvjp(
        lambda x, y: (torch.ge(x, y), []),
        lambda g, res: (None, None)
    )

    return _ge(x, y)


def less(x, y):
    @primitive
    def _less(x, y):
        return torch.lt(x, y)

    _less.defvjp(
        lambda x, y: (torch.lt(x, y), []),
        lambda g, res: (None, None)
    )

    return _less(x, y)


def less_equal(x, y):
    @primitive
    def _le(x, y):
        return torch.le(x, y)

    _le.defvjp(
        lambda x, y: (torch.le(x, y), []),
        lambda g, res: (None, None)
    )

    return _le(x, y)

def equal(x, y):
    @primitive
    def _eq(x, y):
        return torch.eq(x, y)

    _eq.defvjp(
        lambda x, y: (torch.eq(x, y), []),
        lambda g, res: (None, None)
    )

    return _eq(x, y)


def not_equal(x, y):
    @primitive
    def _ne(x, y):
        return torch.ne(x, y)

    _ne.defvjp(
        lambda x, y: (torch.ne(x, y), []),
        lambda g, res: (None, None)
    )

    return _ne(x, y)

def where(condition, x, y):
    @primitive
    def _where(condition, x, y):
        return torch.where(condition, x, y)
    
    _where.defvjp(
    lambda condition, x, y: (torch.where(condition, x, y), [condition]),
    lambda g, res: (
        None,
        where(res[0], g, torch.zeros_like(g)),
        where(res[0], torch.zeros_like(g), g)
    )
)


    return _where(condition, x, y)


def squeeze(x, axis):
    @primitive
    def _squeeze(x):
        return torch.squeeze(x, axis)
    
    _squeeze.defvjp(
        lambda x: (torch.squeeze(x, axis), [axis]),
        lambda g, res: g.unsqueeze(res[0])
    )
    return _squeeze(x)

def unsqueeze(x, axis):
    @primitive
    def _unsqueeze(x):
        return torch.unsqueeze(x, axis)
    
    _unsqueeze.defvjp(
        lambda x: (torch.unsqueeze(x, axis), [axis]),
        lambda g, res: g.squeeze(res[0])
    )
    return _unsqueeze(x)


def cat(xs, axis=0):
    @primitive
    def _cat(*xs):
        return torch.cat(xs, dim=axis)

    sizes = [x.shape[axis] for x in xs]

    _cat.defvjp(
        lambda *xs: (torch.cat(xs, dim=axis), [sizes]),
        lambda g, res: tuple(
            g.narrow(axis, sum(res[0][:i]), res[0][i])
            for i in range(len(res[0]))
        )
    )

    return _cat(*xs)


def stack(xs, axis=0):
    @primitive
    def _stack(*xs):
        return torch.stack(xs, dim=axis)

    _stack.defvjp(
        lambda *xs: (torch.stack(xs, dim=axis), [axis]),
        lambda g, res: tuple(
            g.select(res[0], i) for i in range(g.shape[res[0]])
        )
    )

    return _stack(*xs)


def broadcast_to(x, shape):
    @primitive
    def _broadcast_to(x):
        return torch.broadcast_to(x, shape)

    _broadcast_to.defvjp(
        lambda x: (torch.broadcast_to(x, shape), [x.shape]),
        lambda g, res: unbroadcast(
            torch.empty(res[0], device=g.device), g
        )
    )

    return _broadcast_to(x)

def maximum(x, y):
    @primitive
    def _maximum(x, y):
        return torch.maximum(x, y)

    _maximum.defvjp(
        lambda x, y: (torch.maximum(x, y), [x >= y]),
        lambda g, res: (
            where(res[0], g, torch.zeros_like(g)),
            where(res[0], torch.zeros_like(g), g),
        )
    )

    return _maximum(x, y)

def minimum(x, y):
    @primitive
    def _minimum(x, y):
        return torch.minimum(x, y)

    _minimum.defvjp(
        lambda x, y: (torch.minimum(x, y), [x <= y]),
        lambda g, res: (
            where(res[0], g, torch.zeros_like(g)),
            where(res[0], torch.zeros_like(g), g),
        )
    )

    return _minimum(x, y)

@primitive
def sign(x):
    return torch.sign(x)

sign.defvjp(
    lambda x: (torch.sign(x), []),
    lambda g, res: (torch.zeros_like(g),)
)


@primitive
def abs(x):
    return torch.abs(x)

abs.defvjp(
    lambda x: (torch.abs(x), [x]),
    lambda g, res: (g * sign(res[0]),)
)

def clamp(x, min=None, max=None):
    @primitive
    def _clamp(x):
        return torch.clamp(x, min=min, max=max)

    _clamp.defvjp(
        lambda x: (torch.clamp(x, min=min, max=max), [x]),
        lambda g, res: (
            g * (
                (res[0] >= (min if min is not None else -torch.inf)) &
                (res[0] <= (max if max is not None else torch.inf))
            ),
        )
    )

    return _clamp(x)


def logical_and(x, y):
    @primitive
    def _and(x, y):
        return torch.logical_and(x, y)

    _and.defvjp(
        lambda x, y: (torch.logical_and(x, y), []),
        lambda g, res: (None, None)
    )

    return _and(x, y)


def logical_or(x, y):
    @primitive
    def _or(x, y):
        return torch.logical_or(x, y)

    _or.defvjp(
        lambda x, y: (torch.logical_or(x, y), []),
        lambda g, res: (None, None)
    )

    return _or(x, y)


def logical_not(x):
    @primitive
    def _not(x):
        return torch.logical_not(x)

    _not.defvjp(
        lambda x: (torch.logical_not(x), []),
        lambda g, res: (None,)
    )

    return _not(x)


def logical_xor(x, y):
    @primitive
    def _xor(x, y):
        return torch.logical_xor(x, y)

    _xor.defvjp(
        lambda x, y: (torch.logical_xor(x, y), []),
        lambda g, res: (None, None)
    )

    return _xor(x, y)
