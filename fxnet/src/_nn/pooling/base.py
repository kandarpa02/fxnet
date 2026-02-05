from ...src.functions import mean, reshape, max
from ...src.functions.im2col import im2col, normalize_padding, _to_tuple, get_out_shape


def _can_use_reshape_pool(k, s, p, d): # UNUSED FOR NOW
    return (
        k == s
        and len(set(k)) == 1
        and all(pi == 0 for pi in p)
        and all(di == 1 for di in d)
    )

def _normalize_nd(padding, dims, name):
    if isinstance(padding, int):
        return [(padding, padding)] * dims

    if isinstance(padding, (tuple, list)):
        assert len(padding) == dims
        return padding

    raise TypeError(f"Invalid {name}: {type(padding)}")


def _pool_nd_im2col(
    x,
    kernel_size,
    stride,
    padding,
    dilation,
    reduce,      # max or mean
):
    dims = x.ndim - 2

    padding = normalize_padding(padding, x.shape, _to_tuple(kernel_size, dims), _to_tuple(stride, dims), _to_tuple(dilation, dims))
    k = _to_tuple(kernel_size, dims)      # tuple[int]
    s = _to_tuple(stride, dims)           # tuple[int]
    d = _to_tuple(dilation, dims)

    cols = im2col(
        x,
        kernel_shape=k,
        stride=s,
        padding=padding,
        dilation=d,
    )

    out_shape = get_out_shape(x, cols, k, s, padding, d)
    # cols shape: (N, C * prod(k), prod(out_shape))

    N, CK, O = cols.shape
    C = x.shape[1]
    K_total = CK // C

    cols = reshape(cols, [N, C, K_total, O])

    y = reduce(cols, axis=2)  # pool over kernel

    return reshape(y, [N, C, *out_shape])

from typing import Any, Union
Im2ColArgs = Union[list[int], tuple[int], int, str]

def max_pool1d(x, kernel_size:Im2ColArgs=2, stride:Im2ColArgs=2, padding:Im2ColArgs='same', dilation:Im2ColArgs=1):
    
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=max,
    )


def avg_pool1d(x, kernel_size:Im2ColArgs=2, stride:Im2ColArgs=2, padding:Im2ColArgs='same', dilation:Im2ColArgs=1):
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=mean,
    )

def max_pool2d(x, kernel_size:Im2ColArgs=2, stride:Im2ColArgs=2, padding:Im2ColArgs='same', dilation:Im2ColArgs=1):
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=max,
    )


def avg_pool2d(x, kernel_size:Im2ColArgs=2, stride:Im2ColArgs=2, padding:Im2ColArgs='same', dilation:Im2ColArgs=1):
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=mean,
    )

def max_pool3d(x, kernel_size:Im2ColArgs=2, stride:Im2ColArgs=2, padding:Im2ColArgs='same', dilation:Im2ColArgs=1):
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=max,
    )


def avg_pool3d(x, kernel_size:Im2ColArgs=2, stride:Im2ColArgs=2, padding:Im2ColArgs='same', dilation:Im2ColArgs=1):
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=mean,
    )
