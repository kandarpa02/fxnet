import torch
from ..core import defvjp, novjp

ONES_LIKE = defvjp(
    lambda x: torch.ones_like(x),
    (lambda g, x: None,)
)
