from ..core import primitive
import torch
from .utils import unbroadcast_f
from .utils_ops import sigmoid


add_vjp = lambda g, x, y: (
    unbroadcast_f(x, lambda a:a)(g), 
    unbroadcast_f(y, lambda a:a)(g)
)

sub_vjp = lambda g, x, y: (
    unbroadcast_f(x, lambda a:a)(g), 
    unbroadcast_f(y, lambda a:-a)(g)
)

mul_vjp = lambda g, x, y: (
    unbroadcast_f(x, lambda a:a*y)(g), 
    unbroadcast_f(y, lambda a:a*x)(g)
) 

div_vjp = lambda g, x, y: (
    unbroadcast_f(x, lambda a: a / y)(g),
    unbroadcast_f(y, lambda a: -a * x / (y * y))(g)
)

pow_vjp = lambda g, x, y: (
    unbroadcast_f(x, lambda a: a * y * (x ** (y - 1)))(g),
    unbroadcast_f(y, lambda a: a * (x ** y) * x.log())(g)
)

neg_vjp = lambda g, x: (
    unbroadcast_f(x, lambda a: -a)(g),
)

exp_vjp = lambda g, x: (
    unbroadcast_f(x, lambda a: a * x.exp())(g),
)

log_vjp = lambda g, x: (
    unbroadcast_f(x, lambda a: a / x)(g),
)

log10_vjp = lambda g, x: (
    unbroadcast_f(x, lambda a: a / x.log()),
)

tanh_vjp = lambda g, x: (
    unbroadcast_f(x, lambda a: a * (1 - x.tanh() ** 2))(g),
)

sigmoid_vjp = lambda g, x: (
    unbroadcast_f(x, lambda a: a * sigmoid(x) * (1 - sigmoid(x)))(g),
)

relu_vjp = lambda g, x: (
    unbroadcast_f(x, lambda a: a * (x > 0).to(x.dtype))(g),
)

maximum_vjp = lambda g, x, y: (
    unbroadcast_f(x, lambda a: a * (x >= y).to(x.dtype))(g),
    unbroadcast_f(y, lambda a: a * (y > x).to(y.dtype))(g),
)

minimum_vjp = lambda g, x, y: (
    unbroadcast_f(x, lambda a: a * (x <= y).to(x.dtype))(g),
    unbroadcast_f(y, lambda a: a * (y < x).to(y.dtype))(g),
)

where_vjp = lambda g, c, x, y: (
    None,
    unbroadcast_f(x, lambda a: a * c.to(x.dtype))(g),
    unbroadcast_f(y, lambda a: a * (~c).to(y.dtype))(g),
)

def sum_vjp(g, x, dim=None, keepdim=False):
    if not keepdim and dim is not None:
        for d in sorted((dim,) if isinstance(dim, int) else dim):
            g = g.unsqueeze(d)
    return (g.expand_as(x),)

def mean_vjp(g, x, dim=None, keepdim=False):
    scale = x.numel() if dim is None else x.shape[dim]
    if not keepdim and dim is not None:
        g = g.unsqueeze(dim)
    return (g.expand_as(x) / scale,)

reshape_vjp = lambda g, x, shape: (g.reshape_as(x),)

def transpose_(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim - 1, -1, -1))
    return x.permute(*axes)

transpose_vjp = lambda g, x, axes: (
    transpose_(x, axes),
)

matmul_vjp = lambda g, x, y: (
    unbroadcast_f(x, lambda a: a @ y.transpose(-1, -2))(g),
    unbroadcast_f(y, lambda a: x.transpose(-1, -2) @ a)(g),
)

# ---------- comparison ----------

equal_vjp = lambda g, x, y: (None, None)
not_equal_vjp = lambda g, x, y: (None, None)

greater_vjp = lambda g, x, y: (None, None)
greater_equal_vjp = lambda g, x, y: (None, None)

less_vjp = lambda g, x, y: (None, None)
less_equal_vjp = lambda g, x, y: (None, None)


# ---------- logical ----------

logical_not_vjp = lambda g, x: (None,)

logical_and_vjp = lambda g, x, y: (None, None)
logical_or_vjp  = lambda g, x, y: (None, None)
logical_xor_vjp = lambda g, x, y: (None, None)


# ---------- logical reductions ----------

logical_all_vjp = lambda g, x, dim=None, keepdim=False: (None,)
logical_any_vjp = lambda g, x, dim=None, keepdim=False: (None,)

# extras
astype_vjp = lambda g, x: None