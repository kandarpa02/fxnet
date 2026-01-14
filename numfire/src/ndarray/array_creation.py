from ...backend import backend as b
from ..array import NDarray
from ...src.DType import DType, normalize_dtype
from typing import Optional

def ones(shape, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().ones(shape, dtype))

def zeros(shape, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().zeros(shape, dtype))

def full(shape, value, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().full(shape, value, dtype))

def ones_like(_data, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().ones_like(_data.np, dtype))

def zeros_like(_data, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().zeros_like(_data.np, dtype))

def full_like(_data, value, dtype=None):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().full_like(_data.np, value, dtype))

def arange(start,
    stop = None,
    step = None,
    dtype= None):
    dtype = normalize_dtype(dtype)

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
    axis = 0,
):
    dtype = normalize_dtype(dtype)
    return NDarray(b.xp().linspace(
        start,
        stop,
        num,
        endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
    ))