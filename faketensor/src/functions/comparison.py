from __future__ import annotations
from .._typing import Array as A
from ..base import function
from ..utils import broadcast_backward
from ...backend.backend import xp
from .primitive_array_ops import squeeze
from .primitive_reduct import max

def equal():
    pass