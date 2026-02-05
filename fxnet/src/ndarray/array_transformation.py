from ...backend import backend as b
from ..array import NDarray
from ...src.DType import DType, normalize_dtype
from typing import Optional
import numpy as np
import torch
from .._typing import TensorLike

def one_hot(labels:TensorLike, num_classes:int, dtype:DType|None=None):
    from .base import array
    return array(torch.nn.functional.one_hot(getattr(labels, '__backend_buffer', labels), num_classes), dtype)