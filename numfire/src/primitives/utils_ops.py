import torch.nn.functional as F
import torch
softmax = F.softmax
sigmoid = F.sigmoid
relu = F.relu
tensor = torch.tensor
asarray = torch.asarray
Unfold = F.unfold
einsum = torch.einsum

T = torch

def _to_tuple(v, dims):
    if isinstance(v, int):
        return (v,) * dims
    return tuple(v)

def get_out_shape(image, cols, kernel_shape, stride, padding, dilation):
    N, CK, L = cols.shape
    C = image.shape[1]
    spatial = image.shape[2:]
    dims = len(spatial)
    kernel_shape = _to_tuple(kernel_shape, dims)
    stride = _to_tuple(stride, dims)
    dilation = _to_tuple(dilation, dims)
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


def convolution_f(x, w, stride=1, padding=0, dilation=1):
    """
    padding has to be normalized
    """

    with T.no_grad():
        dims = x.ndim - 2

        stride = _to_tuple(stride, dims)
        dilation = _to_tuple(dilation, dims)

        N, C_in = x.shape[:2]
        C_out = w.shape[0]

        assert w.shape[1] == C_in, (
            f"Expected w.shape[1] == {C_in}, got {w.shape[1]}"
        )

        kernel_shape = w.shape[2:]

        cols = Unfold(
            x,
            kernel_shape,
            dilation,
            padding,
            stride
        )

        out_shape = get_out_shape(
            x, cols, kernel_shape, stride, padding, dilation
        )

        W_col = T.reshape(w, [C_out, -1])
        out = T.einsum("oc,ncp->nop", W_col, cols)
        out = T.reshape(out, [N, C_out, *out_shape])

    return out

def flip_transpose(w):
    """
    w: (C_out, C_in, k1, k2, ..., kD)
    returns:
       (C_in, C_out, k1, k2, ..., kD) with spatial flip
    """
    dims = w.ndim - 2

    w_t = T.swapaxes(w, 0, 1)

    for i in range(dims):
        w_t = T.flip(w_t, dims=(2 + i,))

    return w_t