from ..array import NDarray
from ..DType import normalize_dtype

def array(x, dtype=None):
    """
    helper function to build NDarray
    """
    _dt = normalize_dtype(dtype)
    return NDarray(x, _dt)