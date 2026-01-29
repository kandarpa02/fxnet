from ...backend import backend as b
from ..base import MakeOP
from .._typing import TensorLike
from .im2col import im2col, get_out_shape
# =========================================
# Helper: normalize ndim args
# =========================================
def _to_tuple(v, dims):
    if isinstance(v, int):
        return (v,) * dims
    return tuple(v)

# =========================================
# padding normalization
# =========================================
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


def flip_transpose(w):
    """
    w: (C_out, C_in, k1, k2, ..., kD)
    returns:
       (C_in, C_out, k1, k2, ..., kD) with spatial flip
    """
    dims = w.ndim - 2


    # 1. swap in/out channels
    w_t = torch.swapaxes(w, 0, 1)

    # 2. flip spatial dimensions
    for i in range(dims):
        w_t = torch.flip(w_t, dims=(2 + i,))

    return w_t

import torch
from .utils import maker, unwrap
# =========================================
# Convolution ND
# =========================================
def convolution(x: TensorLike, w: TensorLike, stride=1, padding=0, dilation=1):
    dims = x.ndim - 2

    stride = _to_tuple(stride, dims)
    dilation = _to_tuple(dilation, dims)

    def _fun(x, w):
        from ..array import as_nd

        N, C_in = x.shape[:2]
        C_out = w.shape[0]

        assert w.shape[1] == C_in, (
            f"Expected w.shape[1] == {C_in}, got {w.shape[1]}"
        )

        kernel_shape = w.shape[2:]

        pad_tuple = normalize_padding(
            padding, x.shape, kernel_shape, stride, dilation
        )

        cols = im2col(
            x, kernel_shape, stride, pad_tuple, dilation
        )
        out_shape = get_out_shape(
            x, cols, kernel_shape, stride, padding, dilation
        )

        # print('w', type(unwrap(w)))
        # print('cols', type(unwrap(cols)))
        W_col = unwrap(w).reshape(C_out, -1)
        out = torch.einsum("oc,ncp->nop", W_col, unwrap(cols))
        out = out.reshape((N, C_out, *out_shape))

        def grad_fn(g):
            w_bt = flip_transpose(unwrap(w))

            backward_stride   = dilation
            backward_dilation = stride
            backward_padding = tuple(
                (
                    (kernel_shape[i] - 1) * dilation[i] - pad_tuple[i][0],
                    (kernel_shape[i] - 1) * dilation[i] - pad_tuple[i][1],
                )
                for i in range(dims)
            )


            dx = convolution(
                g,
                w_bt,
                stride=backward_stride,
                padding=backward_padding,
                dilation=backward_dilation
            )
            
            x_ = as_nd(unwrap(x).transpose(0, 1))
            g_ = as_nd(unwrap(g).transpose(0, 1))

            dW = convolution(
                x_,
                g_,
                stride=stride,
                padding=padding,
                dilation=dilation
            )

            return dx, dW

        return as_nd(out), (as_nd(x), as_nd(w)), grad_fn

    return MakeOP(_fun)(x, w)
