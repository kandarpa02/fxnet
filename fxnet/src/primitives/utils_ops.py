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


def compute_padding(padding, x_shape, kernel, stride, dilation):
    if isinstance(padding, int):
        return [(padding, padding)] * len(kernel)

    if isinstance(padding, (tuple, list)) and isinstance(padding[0], int):
        return [(p, p) for p in padding]

    if isinstance(padding, str):
        padding = padding.lower()
        spatial = x_shape[2:]
        pads = []

        for i in range(len(kernel)):
            k_eff = dilation[i] * (kernel[i] - 1) + 1

            if padding == "valid":
                pads.append((0, 0))

            elif padding == "same":
                out = (spatial[i] + stride[i] - 1) // stride[i]
                total = max(0, (out - 1) * stride[i] + k_eff - spatial[i])
                l = total // 2
                r = total - l
                pads.append((l, r))

            elif padding == "full":
                p = k_eff - 1
                pads.append((p, p))

            else:
                raise ValueError(padding)

        return pads

    return padding

def pad_input(x, pads):
    flat = []
    for l, r in reversed(pads):
        flat.extend([l, r])
    return F.pad(x, flat)


# primitive(pad)
# primitive(unfold)
# primitive(fold)
# primitive(einsum)
# primitive(reshape/view)
# primitive(transpose/permute)
# primitive(slice)


def convolution_f(x, w, stride=1, padding="valid", dilation=1):
    dims = x.ndim - 2
    stride   = _to_tuple(stride, dims)
    dilation = _to_tuple(dilation, dims)

    kernel = w.shape[2:]
    pads = compute_padding(padding, x.shape, kernel, stride, dilation)

    x_pad = pad_input(x, pads)

    cols = F.unfold(
        x_pad,
        kernel_size=kernel,
        dilation=dilation,
        padding=0,        # IMPORTANT
        stride=stride,
    )

    W_col = w.reshape(w.shape[0], -1)
    out = torch.einsum("oc,ncl->nol", W_col, cols)

    # compute output shape from padded input
    Hout = (x_pad.shape[2] - dilation[0]*(kernel[0]-1) - 1)//stride[0] + 1
    Wout = (x_pad.shape[3] - dilation[1]*(kernel[1]-1) - 1)//stride[1] + 1

    return out.reshape(x.shape[0], w.shape[0], Hout, Wout), pads

