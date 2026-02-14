from .core import primitive
import torch
from ..DType import DTypeLike, dtype_f

ShapeLike = list[int] | tuple[int] | list | tuple | int 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def _shape(shape):
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def uniform(
    shape: ShapeLike=[],
    low: float = 0.0,
    high: float = 1.0,
    dtype: DTypeLike = None,
    device=None,
):
    d = dtype_f(dtype) if dtype is not None else None

    out = torch.rand(size=_shape(shape), dtype=d, device=DEVICE)

    if low != 0.0 or high != 1.0:
        out = low + (high - low) * out

    return primitive(lambda x: x)(out)


def normal(
    shape: ShapeLike=[],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: DTypeLike = None,
):
    d = dtype_f(dtype) if dtype is not None else None

    out = torch.randn(size=_shape(shape), dtype=d, device=DEVICE)

    if mean != 0.0 or std != 1.0:
        out = mean + std * out

    return primitive(lambda x: x)(out)


def bernoulli(
    shape: ShapeLike=[],
    p: float = 0.5,
    dtype: DTypeLike = None,
):
    d = dtype_f(dtype) if dtype is not None else None

    probs = torch.full(_shape(shape), p, device=DEVICE) 
    out = torch.bernoulli(probs)

    if d is not None:
        out = out.to(d)

    return primitive(lambda x: x)(out)


def randint(
    low: int,
    high: int,
    shape: ShapeLike=[],
    dtype: DTypeLike = None,
):
    d = dtype_f(dtype) if dtype is not None else None

    out = torch.randint(
        low,
        high,
        _shape(shape),
        dtype=d,
        device=DEVICE,
    )

    return primitive(lambda x: x)(out)
