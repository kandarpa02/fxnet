import torch

class DType:
    def __init__(self, base: torch.dtype):
        self.base = base          # the real torch dtype
        self.name = base.__str__()

    def __repr__(self):
        return "fxnet."+f"{self.name}".removeprefix('torch.')
    
    __str__ = __repr__

    def __call__(self): return self.base

DTypeLike = DType|torch.dtype|int|float|complex


float64 = DType(torch.float64)
float32 = DType(torch.float32)
float16 = DType(torch.float16)

int64   = DType(torch.int64)
int32   = DType(torch.int32)
int16   = DType(torch.int16)

bool = DType(torch.bool)


def dtype_f(dt):
    if isinstance(dt, DType):
        return dt()
    if isinstance(dt, (int, float, complex)):
        return dt
    return dt
    