from ..core import primitive
from .utils import unbroadcast
import torch


@primitive
def add(x, y):
    return torch.add(x, y)

add.defvjp(
    lambda x, y: (torch.add(x, y), [x, y]),
    lambda g, res: (
        unbroadcast(res[0], g),
        unbroadcast(res[1], g)
    )
)


@primitive
def mul(x, y):
    return torch.mul(x, y)

mul.defvjp(
    lambda x, y: (torch.mul(x, y), [y, x]),
    lambda g, res: (
        unbroadcast(res[0], g * res[0]),
        unbroadcast(res[1], g * res[1])
    )
)
