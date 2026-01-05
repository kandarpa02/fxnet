from ...backend import backend as b
from ..array import NDarray
from ...src.DType import DType, normalize_dtype
from typing import Optional

def one_hot(labels, num_classes, dtype=None):
    from .array_creation import zeros
    out = zeros(labels.shape + (num_classes,), dtype=dtype)
    valid = (labels >= 0) & (labels < num_classes)
    out[valid, labels[valid]] = 1
    return out

