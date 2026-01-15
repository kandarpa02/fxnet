from ...backend import backend as b
from ..array import NDarray
from ...src.DType import DType, normalize_dtype
from typing import Optional
import numpy as np

def one_hot(labels, num_classes:int, dtype:DType|None=None):
    from .array_creation import zeros
    from ...src.functions.primitive_array_ops import reshape

    flat = reshape(labels, -1)
    out = zeros((flat.shape[0], num_classes), dtype=dtype)

    rows = np.arange(flat.shape[0])
    out[rows, flat] = 1

    return reshape(out, labels.shape + (num_classes,))


