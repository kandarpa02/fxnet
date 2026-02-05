from ...backend.backend import xp
from .utils import maker
import torch

lib = xp()

def argmax(x, axis=None, keepdims=False):
    return maker(x, func=torch.argmax(x, dim=axis, keepdim=keepdims))

def argmin(x, axis=None, keepdims=False):
    return maker(x, func=torch.argmin(x, dim=axis, keepdim=keepdims))