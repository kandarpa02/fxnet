class DType:
    def __init__(self, name):
        self.name = name

    def to_native(self, xp):
        return getattr(xp, self.name)

float16 = DType("float16")
float32 = DType("float32")
float64 = DType("float64")
int16    = DType("int16")
int32    = DType("int32")
int64    = DType("int64")
bool_    = DType("bool_")


from ..backend import backend as b

def normalize_dtype(dtype):
    xp = b.xp()

    if dtype is None:
        return None
    
    # If already xp dtype (numpy/cupy)
    if hasattr(dtype, 'kind'):  
        return dtype

    # NumPy shorthand boolean
    if dtype == '?':
        return bool_.to_native(xp)

    # If our abstract DType
    if isinstance(dtype, DType):
        return dtype.to_native(xp)

    # If string passed
    if isinstance(dtype, str):
        return getattr(xp, dtype)

    raise TypeError(f"Invalid dtype: {dtype}")
