

# 2. AFTER patching → now import jit API
from .src.jit.api import jit

# 3. Rest of imports (order no longer matters)
from .src.autograd.backward import grad, value_and_grad
from .src import autograd
from .src.base import function, no_record
from .src.functions import *
from .src._typing import Array
from . import nn
from .nn.parameters import Variable
from .src.tree_util import register_tree_node, flatten_pytree, unflatten_pytree
from . import optimizers
