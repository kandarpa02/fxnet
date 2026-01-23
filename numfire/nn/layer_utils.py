from .base import Cell
from .parameters import Variable
from .initializers import VarianceScaling, Constant
from ..src.DType import float32, DType, normalize_dtype
from ..src.functions.primitive_arithmetic_and_basic_ops import matmul
from ..backend.backend import xp
from ..src.ndarray.base import array
from ..src.functions.xpy_utils import device_shift

class Input(Cell):
	def __init__(self, input_shape:list[int]|int, dtype:DType=float32, device:str|None=None, name: str | None = None):
		super().__init__(name)
		_shape = input_shape if isinstance(input_shape, list|tuple) else list((input_shape,))
		self._init_array = array(xp().ones(_shape), dtype=dtype)
		if not device is None:
			self._init_array = device_shift(self._init_array, device=device)

	def call(self, model_instance:Cell):
		_ = model_instance(self._init_array)
		return model_instance