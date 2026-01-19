from ..array import NDarray
from ..DType import normalize_dtype
from ...src.functions.xpy_utils import get_dev, module, device_shift
from ...backend.backend import xp

def array(x, dtype=None):
    """
    helper function to build NDarray
    """
    _dt = normalize_dtype(dtype)
    buff = getattr(x, '__backend_buffer__', x)
    return NDarray(xp().array(buff), _dt)