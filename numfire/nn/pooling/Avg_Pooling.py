from .base import avg_pool1d, avg_pool2d, avg_pool3d
from ..base import Cell
from ..parameters import Variable
from ..initializers import VarianceScaling, Constant
from ...src.DType import float32
from ...src.functions.primitive_arithmetic_and_basic_ops import matmul
from typing import Callable

from typing import Any, Union
Im2ColArgs = Union[list[int], tuple[int], int, str]

class AvgPool1D(Cell):
	def __init__(
			self,
			kernel_size:Im2ColArgs=2,
			stride:int|tuple[int]|list[int]=2,
			padding:Im2ColArgs='same',
			name: str | None = None
			):
		super().__init__(name)
		self.pool_fn = lambda x: avg_pool1d(x, kernel_size=kernel_size, stride=stride, padding=padding)

	def call(self, x):
		return self.pool_fn(x)
	
class AvgPool2D(Cell):
	def __init__(
			self,
			kernel_size:Im2ColArgs=2,
			stride:int|tuple[int]|list[int]=2,
			padding:Im2ColArgs='same',
			name: str | None = None
			):
		super().__init__(name)
		self.pool_fn = lambda x: avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
	
	def call(self, x):
		return self.pool_fn(x)
	
class AvgPool3D(Cell):
	def __init__(
			self,
			kernel_size:Im2ColArgs=2,
			stride:int|tuple[int]|list[int]=2,
			padding:Im2ColArgs='same',
			name: str | None = None
			):
		super().__init__(name)
		self.pool_fn = lambda x: avg_pool3d(x, kernel_size=kernel_size, stride=stride, padding=padding)
	
	def call(self, x):
		return self.pool_fn(x)
	
