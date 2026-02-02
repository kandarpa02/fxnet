import torch
from typing import Callable, Any

def metadata(_x):
    x = getattr(_x, '__backend_buffer__', _x)
    shape = x.shape
    ndim = x.ndim
    dtype = x.dtype
    is_complex = x.is_complex()
    return shape, ndim, dtype, is_complex

def unbroadcast(_x, target_meta, broadcast_idx=0):
    x = getattr(_x, '__backend_buffer__', _x)
    target_shape, target_ndim, dtype, target_iscomplex = target_meta
    while x.ndim > target_ndim:
        x = torch.sum(x, dim=broadcast_idx)
    for axis, size in enumerate(target_shape):
        if size == 1:
            x = torch.sum(x, dim=axis, keepdim=True)
    if torch.is_complex(x) and not target_iscomplex:
        x = torch.real(x)
    return x

def unbroadcast_f(target, f) -> Callable[..., Any]:
    target_meta = metadata(target)
    return lambda g: unbroadcast(f(g), target_meta)

