from ...backend import backend as b
from ..base import MakeOP
from .._typing import TensorLike
from .xpy_utils import get_dev, module
import torch
from torch import zeros, arange, stack, meshgrid, broadcast_to, add

def _to_tuple(v, dims):
    if isinstance(v, int):
        return (v,) * dims
    return tuple(v)

def normalize_padding(padding, x_shape, kernel_shape, stride, dilation):
    dims = len(kernel_shape)

    # int symmetric
    if isinstance(padding, int):
        return [(padding, padding)] * dims

    # tuple / list
    if isinstance(padding, (tuple, list)):
        # tuple[int] symmetric
        if all(isinstance(p, int) for p in padding):
            if len(padding) != dims:
                raise ValueError("Padding length must match kernel dims")
            return [(p, p) for p in padding]

        # tuple[tuple] already normalized
        if all(
            isinstance(p, (tuple, list)) and len(p) == 2
            for p in padding
        ):
            if len(padding) != dims:
                raise ValueError("Padding length must match kernel dims")
            return [tuple(p) for p in padding]

    # string modes
    if isinstance(padding, str):
        padding = padding.lower()
        spatial = x_shape[2:]

        if padding == "valid":
            return [(0, 0)] * dims

        if padding == "same":
            pads = []
            for i in range(dims):
                eff_k = dilation[i] * (kernel_shape[i] - 1) + 1
                out = (spatial[i] + stride[i] - 1) // stride[i]
                total = max(0, (out - 1) * stride[i] + eff_k - spatial[i])
                l = total // 2
                r = total - l
                pads.append((l, r))
            return pads

        if padding == "full":
            return [
                (dilation[i] * (kernel_shape[i] - 1),) * 2
                for i in range(dims)
            ]

    raise ValueError(f"Invalid padding: {padding}")


def apply_padding(x, padding):
    pad = []
    for p in reversed(padding):
        if isinstance(p, (tuple, list)):
            pad.extend(p)
        else:
            pad.extend([p, p])
    return torch.nn.functional.pad(x, pad)

import torch
import torch.nn.functional as F
from ..base import MakeOP
from .._typing import TensorLike


def im2col(x: TensorLike, kernel_shape, stride, padding, dilation):
    dims = x.ndim-1 #type:ignore
    kernel_shape = _to_tuple(kernel_shape, dims)
    stride = _to_tuple(stride, dims)
    dilation = _to_tuple(dilation, dims)

    # padding = [(l, r), ...] → torch wants left padding only
    _padding = tuple(p[0] for p in padding)

    def fun(_x):
        from ..array import as_nd
        x = getattr(_x, "__backend_buffer__", _x)

        with torch.no_grad():
            cols = F.unfold(
                x,
                kernel_size=kernel_shape,
                dilation=dilation,
                padding=_padding,
                stride=stride,
            )

        # infer out_shape (same as torch does)
        # N, CK, L = cols.shape
        # C = x.shape[1]
        # spatial = x.shape[2:]
        # dims = len(spatial)


        # out_shape = []
        # # print('L', L)
        # rem = L
        # for i in reversed(range(dims)):
        #     out_i = (
        #         (spatial[i]
        #          + 2 * _padding[i]
        #          - dilation[i] * (kernel_shape[i] - 1) - 1)
        #         // stride[i] + 1
        #     )
        #     out_shape.insert(0, out_i)
        #     rem //= out_i

        # assert rem == 1, (
        #     f"im2col shape mismatch: inferred out_shape={out_shape}, "
        #     f"but unfold produced L={L}"
        # )

        def grad_fn(g):
            return (
                col2im(
                    g,
                    x_shape=x.shape,
                    kernel_shape=kernel_shape,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
            )

        return as_nd(cols), (as_nd(x),), grad_fn

    return MakeOP(fun)(x)

def get_out_shape(image, cols, kernel_shape, stride, padding, dilation):
    N, CK, L = cols.shape
    C = image.shape[1]
    spatial = image.shape[2:]
    dims = len(spatial)
    kernel_shape = _to_tuple(kernel_shape, dims)
    stride = _to_tuple(stride, dims)
    dilation = _to_tuple(dilation, dims)

    if isinstance(padding, str):
        padding = normalize_padding(padding, image.shape, kernel_shape, stride, dilation)
    _padding = tuple(p[0] for p in padding)

    out_shape = []
    rem = L
    for i in reversed(range(dims)):
        out_i = (
            (spatial[i]
                + 2 * _padding[i]
                - dilation[i] * (kernel_shape[i] - 1) - 1)
            // stride[i] + 1
        )
        out_shape.insert(0, out_i)
        rem //= out_i

    assert rem == 1, (
        f"im2col shape mismatch: inferred out_shape={out_shape}, "
        f"but unfold produced L={L}"
    )
    return out_shape


def col2im(
    cols: TensorLike,
    x_shape,
    kernel_shape,
    stride,
    padding,
    dilation,
):
    dims = cols.ndim-1
    kernel_shape = _to_tuple(kernel_shape, dims)
    stride = _to_tuple(stride, dims)
    dilation = _to_tuple(dilation, dims)

    torch_padding = tuple(p[0] for p in padding)
    output_size = x_shape[2:]

    def fun(_cols):
        from ..array import as_nd
        cols = getattr(_cols, "__backend_buffer__", _cols)

        with torch.no_grad():
            dx = F.fold(
                cols,
                output_size=output_size,
                kernel_size=kernel_shape,
                dilation=dilation,
                padding=torch_padding,
                stride=stride,
            )

        def grad_fn(g):

            return (
                im2col(
                    g,
                    kernel_shape=kernel_shape,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
            )

        return as_nd(dx), (as_nd(cols),), grad_fn

    return MakeOP(fun)(cols)
