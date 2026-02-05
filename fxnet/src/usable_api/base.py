from ..DType import normalize_dtype, DType
import torch
from .._typing import TensorLike
from ..tensor_value import TensorBox

DtypeLike = DType|str|None

def array(x, dtype=None):
    """
    helper function to build NDarray
    """
    _dt = normalize_dtype(dtype)
    buff = getattr(x, '__backend_buffer__', x)
    return NDarray(buff, _dt)

def constant(x: TensorLike, dtype: DtypeLike = None):
    """
    Create an immutable tensor with a fixed value.

    Constants participate fully in automatic differentiation, but are
    immutable and are not considered model parameters. As a result,
    constants are ignored by optimizers and do not have an identity
    (such as a name or trainable flag).

    This function is intended for inputs, targets, masks, and other
    fixed values where gradient flow is required but parameter updates
    are not.

    Parameters
    ----------
    x : TensorLike
        Initial value of the tensor. Can be a Python scalar, list, tuple,
        NumPy array, or compatible tensor-like object.
    dtype : DtypeLike, optional
        Data type of the tensor. If None, the dtype is inferred from `x`.

    Returns
    -------
    Tensor
        An immutable tensor supporting automatic differentiation.

    Examples
    --------
    >>> a = nf.constant([1., 3, 5, 6], nf.float32)
    >>> print(a)
    Tensor([1., 3., 5., 6.])
    """
    _dt = normalize_dtype(dtype)
    return TensorBox(x, dtype=_dt)