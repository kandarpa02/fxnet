from ...backend import backend as b
from ...src.DType import DType, normalize_dtype
from typing import Optional, Any
from ..functions.xpy_utils import module, get_dev

def ones(shape, dtype=None, device:Any=None):
    dtype = normalize_dtype(dtype)
    from ..array import NDarray
    if device is not None:
        return NDarray(module(device).ones(shape=shape, dtype=dtype))
    return NDarray(b.xp().ones(shape, dtype))

def zeros(shape, dtype=None, device:Any=None):
    dtype = normalize_dtype(dtype)
    from ..array import NDarray
    if device is not None:
        return NDarray(module(device).zeros(shape=shape, dtype=dtype))
    return NDarray(b.xp().zeros(shape, dtype))

def full(shape, value, dtype=None, device:Any=None):
    dtype = normalize_dtype(dtype)
    from ..array import NDarray
    if device is not None:
        return NDarray(module(device).full(shape=shape, fill_value=value, dtype=dtype))
    return NDarray(b.xp().full(shape=shape, fill_value=value, dtype=dtype))

def ones_like(_data, dtype=None, device:Any=None):
    dtype = normalize_dtype(dtype)
    from ..array import NDarray
    if device is not None:
        return NDarray(module(device).ones_like(getattr(_data, '__backend_buffer__', _data), dtype=dtype))
    d = get_dev(_data)
    return NDarray(module(d).ones_like(getattr(_data, '__backend_buffer__', _data), dtype=dtype))

def zeros_like(_data, dtype=None, device:Any=None):
    dtype = normalize_dtype(dtype)
    from ..array import NDarray
    if device is not None:
        return NDarray(module(device).zeros_like(getattr(_data, '__backend_buffer__', _data), dtype=dtype))
    d = get_dev(_data)
    return NDarray(module(d).zeros_like(getattr(_data, '__backend_buffer__', _data), dtype=dtype))

def full_like(_data, value, dtype=None, device:Any=None):
    dtype = normalize_dtype(dtype)
    from ..array import NDarray
    if device is not None:
        return NDarray(module(device).full_like(getattr(_data, '__backend_buffer__', _data), fill_value=value, dtype=dtype))
    d = get_dev(_data)
    return NDarray(module(d).zeros_like(getattr(_data, '__backend_buffer__', _data), dtype=dtype))

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