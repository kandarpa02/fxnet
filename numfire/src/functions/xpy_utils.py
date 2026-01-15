def get_dev(*arrays):
    """
    Determine the device of the given arrays.

    Priority:
        - If any array is a CuPy array -> 'cuda'
        - Otherwise, if any array is a NumPy array -> 'cpu'
        - Otherwise -> None

    Supports:
        - Raw np.ndarray / cp.ndarray
        - NDarray wrappers (via __backend_buffer__)
    """
    import numpy as np
    try:
        import cupy as cp
        has_cupy = True
    except ImportError:
        cp = None
        has_cupy = False

    # First, check for GPU arrays
    if has_cupy:
        for arr in arrays:
            buf = getattr(arr, "__backend_buffer__", arr)
            if isinstance(buf, cp.ndarray):
                return "cuda"

    # Then, check for CPU arrays
    for arr in arrays:
        buf = getattr(arr, "__backend_buffer__", arr)
        if isinstance(buf, np.ndarray):
            return "cpu"

    return 'cpu'
