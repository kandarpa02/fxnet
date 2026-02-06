import torch

class DType:
    def __init__(self, name):
        self.name = name

    def native(self):
        return getattr(torch, self.name)
    
    def __repr__(self) -> str:
        return f"numfire.{self.name}"

    __str__ = __repr__

    @classmethod
    def from_torch_dtype(cls, dtype:str):
        return DType(dtype.removeprefix('torch.'))

float16 = DType("float16")
float32 = DType("float32")
float64 = DType("float64")
int16    = DType("int16")
int32    = DType("int32")
int64    = DType("int64")
bool_    = DType("bool")

def dname(d):
	f = str(d).removeprefix("<class '").removesuffix("'>")
	dmap = {'float':float32, 'bool':bool_, 'int':int32}
	return dmap.get(f, float32)

def normalize_dtype(dtype, string=False) -> torch.dtype:
    if dtype is None:
        return None
    
    # If already xp dtype (numpy/cupy)
    if hasattr(dtype, 'kind'):  
        return dtype

    # NumPy shorthand boolean
    if dtype == '?':
        if string: 
            return bool_.name
        return bool_.native()

    # If our abstract DType
    if isinstance(dtype, DType):
        if string: 
            return dtype.name
        return dtype.native()
    
    if isinstance(dtype, type):
        if string: 
            return dname(dtype).name
        return dname(dtype).native()

    # If string passed
    if isinstance(dtype, str):
        if string: 
            return getattr(torch, dtype).__repr__()
        return getattr(torch, dtype)

    raise TypeError(f"Invalid dtype: {dtype}")

def reverse_dtype(dtype)->str:
    return DType.from_torch_dtype(dtype=dtype).__repr__()