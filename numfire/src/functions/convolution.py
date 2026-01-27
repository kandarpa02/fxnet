from ...backend import backend as b
from ..base import MakeOP
from .._typing import Array

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

    if isinstance(padding, int):
        return (padding,) * dims
    if isinstance(padding, (tuple, list)):
        return tuple(padding)

    if not isinstance(padding, str):
        raise ValueError("padding must be int/tuple/string")

    padding = padding.lower()
    spatial = x_shape[2:]

    if padding == "valid":
        return (0,) * dims

    if padding == "same":
        pads = []
        for i in range(dims):
            in_dim = spatial[i]
            k = kernel_shape[i]
            d = dilation[i]
            s = stride[i]

            eff_k = d * (k - 1) + 1
            out_dim = (in_dim + s - 1) // s
            total = max(0, (out_dim - 1) * s + eff_k - in_dim)
            pads.append(total // 2)
        return tuple(pads)

    if padding == "full":
        return tuple(dilation[i] * (kernel_shape[i] - 1) for i in range(dims))

    raise ValueError(f"Unknown padding mode: {padding}")


# =========================================
# im2col ND (FIXED)
# =========================================
from .xpy_utils import get_dev, module

def im2col_nd(x, kernel_shape, stride, padding, dilation):
    d = get_dev(x)
    pad = module(d).pad
    arange = module(d).arange
    stack = module(d).stack
    meshgrid = module(d).meshgrid
    broadcast_to = module(d).broadcast_to

    N, C = x.shape[:2]
    spatial = x.shape[2:]
    dims = len(spatial)

    pad_width = [(0, 0), (0, 0)] + [(p, p) for p in padding]
    xpad = pad(x, pad_width)

    out_shape = [
        (spatial[i] + 2 * padding[i]
         - dilation[i] * (kernel_shape[i] - 1) - 1) // stride[i] + 1
        for i in range(dims)
    ]

    # kernel offsets
    k_list = [arange(kernel_shape[i]) * dilation[i] for i in range(dims)]
    k_grid = stack(meshgrid(*k_list, indexing="ij"), axis=0)
    k_grid = k_grid.reshape(dims, -1, 1)  # (dims, K, 1)

    # window offsets
    w_list = [arange(out_shape[i]) * stride[i] for i in range(dims)]
    w_grid = stack(meshgrid(*w_list, indexing="ij"), axis=0)
    w_grid = w_grid.reshape(dims, 1, -1)  # (dims, 1, O)

    idx = k_grid + w_grid  # (dims, K, O)

    K_total = idx.shape[1]
    O_total = idx.shape[2]

    # broadcast indices
    N_idx = arange(N).reshape(N, 1, 1, 1)
    C_idx = arange(C).reshape(1, C, 1, 1)

    N_idx = broadcast_to(N_idx, (N, C, K_total, O_total))
    C_idx = broadcast_to(C_idx, (N, C, K_total, O_total))

    full_idx = [N_idx, C_idx]
    for d in range(dims):
        full_idx.append(
            broadcast_to(idx[d][None, None, :, :],
                            (N, C, K_total, O_total))
        )

    patches = xpad[tuple(full_idx)]  # (N, C, K, O)
    cols = patches.reshape(N, C * K_total, O_total)

    return cols, out_shape


# =========================================
# col2im ND (FIXED)
# =========================================
def col2im_nd(cols, x_shape, kernel_shape, stride, padding, dilation, out_shape):
    d = get_dev(cols)
    zeros = module(d).zeros
    arange = module(d).arange
    stack = module(d).stack
    meshgrid = module(d).meshgrid
    broadcast_to = module(d).broadcast_to
    add = module(d).add

    N, C = x_shape[:2]
    dims = len(kernel_shape)
    spatial = x_shape[2:]

    padded_shape = (N, C) + tuple(
        spatial[i] + 2 * padding[i] for i in range(dims)
    )
    xpad = zeros(padded_shape, dtype=cols.dtype)

    cols_rs = cols.reshape(
        (N, C) + tuple(kernel_shape) + tuple(out_shape)
    )

    k_list = [arange(kernel_shape[i]) * dilation[i] for i in range(dims)]
    k_grid = stack(meshgrid(*k_list, indexing="ij"), axis=0)
    k_grid = k_grid.reshape(dims, -1, 1)

    w_list = [arange(out_shape[i]) * stride[i] for i in range(dims)]
    w_grid = stack(meshgrid(*w_list, indexing="ij"), axis=0)
    w_grid = w_grid.reshape(dims, 1, -1)

    idx = k_grid + w_grid
    K_total, O_total = idx.shape[1], idx.shape[2]

    N_idx = arange(N).reshape(N, 1, 1, 1)
    C_idx = arange(C).reshape(1, C, 1, 1)

    N_idx = broadcast_to(N_idx, (N, C, K_total, O_total))
    C_idx = broadcast_to(C_idx, (N, C, K_total, O_total))

    full_idx = [N_idx, C_idx]
    for d in range(dims):
        full_idx.append(
            broadcast_to(idx[d][None, None, :, :],
                            (N, C, K_total, O_total))
        )

    add.at(
        xpad,
        tuple(full_idx),
        cols_rs.reshape(N, C, K_total, O_total)
    )

    slices = [slice(None), slice(None)] + [
        slice(padding[i], padding[i] + spatial[i]) for i in range(dims)
    ]
    return xpad[tuple(slices)]

def conv_backward_input_nd(g, w, x_shape, stride, padding, dilation):
    d = get_dev(g)
    zeros = module(d).zeros

    N, C_out = g.shape[:2]
    C_in = w.shape[1]
    dims = w.ndim - 2

    spatial = x_shape[2:]
    dx = zeros(x_shape, dtype=g.dtype)

    kernel_shape = w.shape[2:]
    out_shape = g.shape[2:]

    # kernel offsets
    k_list = [module(d).arange(kernel_shape[i]) * dilation[i]
              for i in range(dims)]
    k_grid = module(d).stack(
        module(d).meshgrid(*k_list, indexing="ij"), axis=0
    ).reshape(dims, -1)  # (dims, K)

    # output positions
    o_list = [module(d).arange(out_shape[i]) * stride[i]
              for i in range(dims)]
    o_grid = module(d).stack(
        module(d).meshgrid(*o_list, indexing="ij"), axis=0
    ).reshape(dims, -1)  # (dims, O)

    K, O = k_grid.shape[1], o_grid.shape[1]

    for k in range(K):
        for o in range(O):
            in_pos = o_grid[:, o] + k_grid[:, k] - padding
            if ((in_pos < 0).any() or
                (in_pos >= spatial).any()):
                continue

            idx = (slice(None), slice(None), *in_pos.tolist())
            dx[idx] += (
                g.reshape(N, C_out, -1)[:, :, o]
                @ w[:, :, k].T
            )

    return dx

def flip_transpose(w):
    """
    w: (C_out, C_in, k1, k2, ..., kD)
    returns:
       (C_in, C_out, k1, k2, ..., kD) with spatial flip
    """
    d = get_dev(w)
    mod = module(d)

    dims = w.ndim - 2

    # 1. swap in/out channels
    w_t = mod.swapaxes(w, 0, 1)

    # 2. flip spatial dimensions
    for i in range(dims):
        w_t = mod.flip(w_t, axis=2 + i)

    return w_t


# =========================================
# Convolution ND
# =========================================
def convolution(x: Array, w: Array, stride=1, padding=0, dilation=1):
    d = get_dev(x)
    einsum = module(d).einsum
    dims = x.ndim - 2

    stride = _to_tuple(stride, dims)
    dilation = _to_tuple(dilation, dims)

    def _fun(x, w):
        from ..array import as_nd
        x, w = as_nd(x), as_nd(w)

        x_np, w_np = x.np, w.np

        N, C_in = x_np.shape[:2]
        C_out = w_np.shape[0]

        assert w_np.shape[1] == C_in, (
            f"Expected w.shape[1] == {C_in}, got {w_np.shape[1]}"
        )

        kernel_shape = w_np.shape[2:]

        pad_tuple = normalize_padding(
            padding, x_np.shape, kernel_shape, stride, dilation
        )

        cols, out_shape = im2col_nd(
            x_np, kernel_shape, stride, pad_tuple, dilation
        )

        W_col = w_np.reshape(C_out, -1)
        out = einsum("oc,ncp->nop", W_col, cols)
        out = out.reshape((N, C_out, *out_shape))

        def grad_fn(g):
            w_bt = flip_transpose(w_np)

            backward_stride   = dilation
            backward_dilation = stride
            backward_padding  = tuple(
                (kernel_shape[i] - 1) * dilation[i] - pad_tuple[i]
                for i in range(dims)
)
            dx = convolution(
                g,
                w_bt,
                stride=backward_stride,
                padding=backward_padding,
                dilation=backward_dilation
            )

            dW = convolution(
                x,
                g,
                stride=stride,
                padding=padding,
                dilation=dilation
            )

            return dx, dW

        return as_nd(out), (x, w), grad_fn

    return MakeOP(_fun)(x, w)
