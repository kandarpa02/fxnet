
# 3. Rest of imports (order no longer matters)
from .data import data
from .src.autograd.backward import grad, value_and_grad
from .src import autograd
from .src.base import no_record
from .src.utils import custom_function
from .src.functions import *
from .src._typing import Array
from .src.DType import (
    DType, int16, int32, int64,
    float16, float32, float64, bool_
)
from . import nn
# from .src.tree_util import register_tree_node, flatten_pytree, unflatten_pytree
from . import tree
from . import optimizers

from .src.ndarray.utils import astype
# from .src.eager_fusion import fuse