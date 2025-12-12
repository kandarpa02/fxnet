import numpy as _np

_device = "cpu"
_xp = _np       # default

def _try_cupy():
    try:
        import cupy as cp
        return cp
    except Exception:
        return None


def set_device(device: str):
    global _device, _xp

    if device == "cpu":
        _device = "cpu"
        _xp = _np
        return

    if device == "cuda":
        cp = _try_cupy()
        if cp is None:
            raise RuntimeError("CUDA requested but CuPy not installed.")
        _device = "cuda"
        _xp = cp
        return

    if device == "auto":
        cp = _try_cupy()
        if cp is not None:
            _device = "cuda"
            _xp = cp
        else:
            _device = "cpu"
            _xp = _np
        return

    raise ValueError(f"Unknown device '{device}'")


# auto detect at import
set_device("auto")


def xp():
    """Return active array module (numpy or cupy)."""
    return _xp


def get_device():
    return _device
