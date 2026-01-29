from ...backend import backend as b
from ...src.DType import DType, normalize_dtype
from typing import Optional, Any
from ..functions.xpy_utils import module, get_dev
from torch import ones, ones_like, zeros, zeros_like
import torch

def full(shape, value, dtype=None, device:Any=None):
    dtype = normalize_dtype(dtype)
    from ..array import NDarray
    return NDarray(torch.full(shape, value, dtype=dtype, device=device))

def full_like(_data, value, dtype=None, device:Any=None):
    return full(_data.shape, value=value, dtype=dtype, device=device)

def ones(shape, dtype=None, device:Any=None):
    return full(shape, 1.0, dtype=dtype, device=device)

def zeros(shape, dtype=None, device:Any=None):
    return full(shape, 0.0, dtype=dtype, device=device)

def ones_like(_data, dtype=None, device:Any=None):
    return full(_data.shape, 1.0, dtype=dtype, device=device)

def zeros_like(_data, dtype=None, device:Any=None):
    return full(_data.shape, 0.0, dtype=dtype, device=device)

def arange(start,
    stop = None,
    step = None,
    dtype= None,
    device:Any=None):
    
    dtype = normalize_dtype(dtype)
    from ..array import NDarray

    if device is not None:
        return NDarray(module(device).arange(start, stop, step, dtype=dtype))
    
    return NDarray(b.xp().arange(
        start=start,
        stop=stop,
        step=step,
        dtype=dtype
    ))

def linespace(
    start,
    stop,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype = None,
    device:Any=None,
    axis = 0,
):
    dtype = normalize_dtype(dtype)
    from ..array import NDarray
    if device is not None:
        return NDarray(module(device).linspace(start, stop, num, endpoint, retstep=retstep, dtype=dtype, axis=axis))
    return NDarray(b.xp().linspace(
        start,
        stop,
        num,
        endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
    ))