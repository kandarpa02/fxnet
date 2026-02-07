import torch
from ..core import defvjp

ONES_LIKE = defvjp(
    lambda x: torch.ones_like(x),
    (lambda g, x: None,)
)
