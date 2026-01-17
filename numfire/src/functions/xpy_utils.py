from ...backend.backend import xp
from xpy.utils import shift_device_
from .._typing import Array

def get_dev(*arrays):
    import numpy as np
    try:
        import cupy as cp
        has_cupy = True
    except ImportError:
        cp = None
        has_cupy = False

    def unwrap(x):
        return getattr(x, "__backend_buffer__", x)

    if has_cupy:
        for arr in arrays:
            buf = unwrap(arr)
            if isinstance(buf, cp.ndarray):
                return "cuda"

    for arr in arrays:
        buf = unwrap(arr)
        if isinstance(buf, np.ndarray):
            return "cpu"

    for arr in arrays:
        if isinstance(arr, (int, float, bool, list)):
            return "cuda" if has_cupy else "cpu"

    return "cuda" if has_cupy else "cpu"

def module(type):
    if type == "cuda":
        try:
            import cupy as cp
            return cp
        except ImportError:
            import numpy as np
            return np

    if type == "cpu":
        import numpy as np
        return np

    raise TypeError(f"Unknown backend type: {type}")

def device(x:Array):
    """
    utility function to get the backend info of the given input
    """
    return get_dev(x)


def device_shift(x: Array, device: str):
    """Shift x to preferred device: 'cpu' or 'cuda'."""
    from ..array import as_nd

    # Safely unwrap NDarray, otherwise keep x as-is
    buf = getattr(x, "__backend_buffer__", x)

    return as_nd(shift_device_(buf, device=device))
