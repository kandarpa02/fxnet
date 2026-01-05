from ..array import as_nd
from ...backend.backend import xp
from .utils import unwrap

lib = xp()

def argmax(x, axis=None, keepdims=False):
    return as_nd(lib.argmax(unwrap(x), axis=axis, keepdims=keepdims))

def argmin(x, axis=None, keepdims=False):
    return as_nd(lib.argmin(unwrap(x), axis=axis, keepdims=keepdims))