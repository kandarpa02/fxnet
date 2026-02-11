from ..core import primitive
from .utils import unbroadcast
import torch
from collections.abc import Sequence

# add
# sub 
# mul
# div
# neg
# pow
# exp
# log
# tanh
# sum
# mean
# matmul

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

pow.defvjp(
    lambda x, y: (torch.pow(x, y), [x, y]),
    lambda g, res: (
        unbroadcast(res[0], g * res[1] * (res[0] ** (res[1] - 1))),
        unbroadcast(res[1], g * res[0].log() * (res[0] ** res[1]))
    )
)


# ---------------- exp ----------------

@primitive
def exp(x):
    return torch.exp(x)

exp.defvjp(
    lambda x: (torch.exp(x), [torch.exp(x)]),
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
        unbroadcast(res[0], g @ res[1].transpose(-1, -2)),
        unbroadcast(res[1], res[0].transpose(-1, -2) @ g),
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
