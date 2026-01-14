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
# PyTorch-like padding normalization
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
def im2col_nd(x, kernel_shape, stride, padding, dilation):
    xp = b.xp()

    N, C = x.shape[:2]
    spatial = x.shape[2:]
    dims = len(spatial)

    pad_width = [(0, 0), (0, 0)] + [(p, p) for p in padding]
    xpad = xp.pad(x, pad_width)

    out_shape = [
        (spatial[i] + 2 * padding[i]
         - dilation[i] * (kernel_shape[i] - 1) - 1) // stride[i] + 1
        for i in range(dims)
    ]

    # kernel offsets
    k_list = [xp.arange(kernel_shape[i]) * dilation[i] for i in range(dims)]
    k_grid = xp.stack(xp.meshgrid(*k_list, indexing="ij"), axis=0)
    k_grid = k_grid.reshape(dims, -1, 1)  # (dims, K, 1)

    # window offsets
    w_list = [xp.arange(out_shape[i]) * stride[i] for i in range(dims)]
    w_grid = xp.stack(xp.meshgrid(*w_list, indexing="ij"), axis=0)
    w_grid = w_grid.reshape(dims, 1, -1)  # (dims, 1, O)

    idx = k_grid + w_grid  # (dims, K, O)

    K_total = idx.shape[1]
    O_total = idx.shape[2]

    # broadcast indices
    N_idx = xp.arange(N).reshape(N, 1, 1, 1)
    C_idx = xp.arange(C).reshape(1, C, 1, 1)

    N_idx = xp.broadcast_to(N_idx, (N, C, K_total, O_total))
    C_idx = xp.broadcast_to(C_idx, (N, C, K_total, O_total))

    full_idx = [N_idx, C_idx]
    for d in range(dims):
        full_idx.append(
            xp.broadcast_to(idx[d][None, None, :, :],
                            (N, C, K_total, O_total))
        )

    patches = xpad[tuple(full_idx)]  # (N, C, K, O)
    cols = patches.reshape(N, C * K_total, O_total)

    return cols, out_shape


# =========================================
# col2im ND (FIXED)
# =========================================
def col2im_nd(cols, x_shape, kernel_shape, stride, padding, dilation, out_shape):
    xp = b.xp()

    N, C = x_shape[:2]
    dims = len(kernel_shape)
    spatial = x_shape[2:]

    padded_shape = (N, C) + tuple(
        spatial[i] + 2 * padding[i] for i in range(dims)
    )
    xpad = xp.zeros(padded_shape, dtype=cols.dtype)

    cols_rs = cols.reshape(
        (N, C) + tuple(kernel_shape) + tuple(out_shape)
    )

    k_list = [xp.arange(kernel_shape[i]) * dilation[i] for i in range(dims)]
    k_grid = xp.stack(xp.meshgrid(*k_list, indexing="ij"), axis=0)
    k_grid = k_grid.reshape(dims, -1, 1)

    w_list = [xp.arange(out_shape[i]) * stride[i] for i in range(dims)]
    w_grid = xp.stack(xp.meshgrid(*w_list, indexing="ij"), axis=0)
    w_grid = w_grid.reshape(dims, 1, -1)

    idx = k_grid + w_grid
    K_total, O_total = idx.shape[1], idx.shape[2]

    N_idx = xp.arange(N).reshape(N, 1, 1, 1)
    C_idx = xp.arange(C).reshape(1, C, 1, 1)

    N_idx = xp.broadcast_to(N_idx, (N, C, K_total, O_total))
    C_idx = xp.broadcast_to(C_idx, (N, C, K_total, O_total))

    full_idx = [N_idx, C_idx]
    for d in range(dims):
        full_idx.append(
            xp.broadcast_to(idx[d][None, None, :, :],
                            (N, C, K_total, O_total))
        )

    xp.add.at(
        xpad,
        tuple(full_idx),
        cols_rs.reshape(N, C, K_total, O_total)
    )

    slices = [slice(None), slice(None)] + [
        slice(padding[i], padding[i] + spatial[i]) for i in range(dims)
    ]
    return xpad[tuple(slices)]


# =========================================
# Convolution ND
# =========================================
def convolution(x: Array, w: Array, stride=1, padding=0, dilation=1):
    xp = b.xp()
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
        out = xp.einsum("oc,ncp->nop", W_col, cols)
        out = out.reshape((N, C_out, *out_shape))

        def grad_fn(g):
            from .primitive_array_ops import reshape
            g2 = reshape(g, [N, C_out, -1])

            dW = xp.einsum("nop,ncp->oc", g2, cols)
            dW = dW.reshape(w_np.shape)

            dCols = xp.einsum("co,nop->ncp", W_col.T, g2)

            dx = col2im_nd(
                dCols, x_np.shape, kernel_shape,
                stride, pad_tuple, dilation, out_shape
            )

            return as_nd(dx), as_nd(dW)

        return as_nd(out), (x, w), grad_fn

    return MakeOP(_fun)(x, w)
