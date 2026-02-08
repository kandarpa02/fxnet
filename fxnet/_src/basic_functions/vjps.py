from ..core import defvjp, novjp
from .utils import unbroadcast
import torch

def add(x, y):
    return defvjp(
        torch.add,
        (
            lambda g, x, y: unbroadcast(x, g),
            lambda g, x, y: unbroadcast(y, g)
        )
    )(x, y)

def mul(x, y):
    return defvjp(
        torch.mul,
        (
            lambda g, x, y: unbroadcast(x, g * y),
            lambda g, x, y: unbroadcast(y, g * x)
        )
    )(x, y)

