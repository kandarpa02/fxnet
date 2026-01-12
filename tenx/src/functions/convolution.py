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
    """
    Supports:
        - int, tuple
        - "valid", "same", "full"
    """
    dims = len(kernel_shape)

    # --------------------------
    # numeric
    # --------------------------
    if isinstance(padding, int):
        return (padding,) * dims
    if isinstance(padding, (tuple, list)):
        return tuple(padding)

    # --------------------------
    # symbolic
    # --------------------------
    if not isinstance(padding, str):
        raise ValueError("padding must be int/tuple/string")

    padding = padding.lower()
    spatial = x_shape[2:]

    # ---------------- VALID -----------------
    if padding == "valid":
        return tuple(0 for _ in range(dims))

    # ---------------- SAME ------------------
    if padding == "same":
        pads = []
        for i in range(dims):
            in_dim = spatial[i]
            k = kernel_shape[i]
            d = dilation[i]
            s = stride[i]

            eff_k = d * (k - 1) + 1
            out_dim = (in_dim + s - 1) // s  # ceil
            total = max(0, (out_dim - 1) * s + eff_k - in_dim)
            pads.append(total // 2)

        return tuple(pads)

    # ---------------- FULL ------------------
    if padding == "full":
        pads = []
        for i in range(dims):
            k = kernel_shape[i]
            d = dilation[i]
            eff_k = d * (k - 1) + 1
            pads.append(eff_k - 1)
        return tuple(pads)

    raise ValueError(f"Unknown padding mode: {padding}")


# =========================================
# FIXED im2col ND
# =========================================
def im2col_nd(x, kernel_shape, stride, padding, dilation):
    xp = b.xp()

    N, C = x.shape[:2]
    spatial = x.shape[2:]
    dims = len(spatial)

    # pad input
    pad_width = [(0,0), (0,0)] + [(p,p) for p in padding]
    xpad = xp.pad(x, pad_width)

    # output spatial shape
    out_shape = [
        (spatial[i] + 2*padding[i]
         - dilation[i] * (kernel_shape[i] - 1) - 1) // stride[i] + 1
        for i in range(dims)
    ]

    # ----------------------------------------------------
    # CORRECT: flat kernel offsets → (dims, K)
    # ----------------------------------------------------
    k_list = []
    for i in range(dims):
        k = xp.arange(kernel_shape[i]) * dilation[i]
        k_list.append(k)
    k_grid = xp.stack(xp.meshgrid(*k_list, indexing="ij"), axis=0)
    k_grid = k_grid.reshape(dims, -1, 1)    # (dims, K, 1)

    # ----------------------------------------------------
    # CORRECT: flat window offsets → (dims, O)
    # ----------------------------------------------------
    w_list = []
    for i in range(dims):
        w = xp.arange(out_shape[i]) * stride[i]
        w_list.append(w)
    w_grid = xp.stack(xp.meshgrid(*w_list, indexing="ij"), axis=0)
    w_grid = w_grid.reshape(dims, 1, -1)    # (dims, 1, O)

    # ----------------------------------------------------
    # BROADCAST CORRECTLY NOW: (dims, K, O)
    # ----------------------------------------------------
    idx = k_grid + w_grid

    # ----------------------------------------------------
    # Build advanced index arrays for xpad
    # ----------------------------------------------------
    N_idx = xp.arange(N)[:, None, None]
    C_idx = xp.arange(C)[None, :, None]

    full_idx = [N_idx, C_idx]
    for d in range(dims):
        full_idx.append(idx[d])

    # patches shape → (N, C, K, O)
    patches = xpad[tuple(full_idx)]

    K_total = int(xp.prod(kernel_shape))
    O_total = int(xp.prod(out_shape))

    cols = patches.reshape(N, C*K_total, O_total)
    return cols, out_shape


# =========================================
# col2im ND
# =========================================
def col2im_nd(cols, x_shape, kernel_shape, stride, padding, dilation, out_shape):
    xp = b.xp()

    N, C = x_shape[:2]
    dims = len(kernel_shape)
    spatial = x_shape[2:]

    padded_shape = (N, C) + tuple(spatial[i] + 2*padding[i] for i in range(dims))
    xpad = xp.zeros(padded_shape, dtype=cols.dtype)

    # reshape cols → (N, C, K1,...,Kd, O1,...,Od)
    cols_rs = cols.reshape((N, C) + tuple(kernel_shape) + tuple(out_shape))

    # build kernel offsets
    k_list = [xp.arange(kernel_shape[i]) * dilation[i] for i in range(dims)]
    k_grid = xp.stack(xp.meshgrid(*k_list, indexing="ij"), axis=0).reshape(dims, -1, 1)

    # build window offsets
    w_list = [xp.arange(out_shape[i]) * stride[i] for i in range(dims)]
    w_grid = xp.stack(xp.meshgrid(*w_list, indexing="ij"), axis=0).reshape(dims, 1, -1)

    idx = k_grid + w_grid  # (dims, K_total, O_total)

    # build advanced index arrays for N and C
    N_idx = xp.arange(N)[:, None, None]
    C_idx = xp.arange(C)[None, :, None]

    # accumulate with np.add.at to handle overlaps
    full_idx = [N_idx, C_idx] + [idx[d] for d in range(dims)]
    np.add.at(xpad, tuple(full_idx), cols_rs.reshape((N, C, -1, idx.shape[2])))

    # unpad
    slices = [slice(None), slice(None)] + [slice(padding[i], padding[i] + spatial[i]) for i in range(dims)]
    return xpad[tuple(slices)]



# =========================================
# Convolution ND  (WITH TORCH-LIKE PADDING)
# =========================================
def convolution(x: Array, w: Array, stride=1, padding=0, dilation=1):
    xp = b.xp()
    dims = x.ndim - 2

    stride = _to_tuple(stride, dims)
    dilation = _to_tuple(dilation, dims)

    def _fun(x, w):
        x_np, w_np = x.np, w.np
        from ..array import as_nd

        N, C_in = x_np.shape[:2]
        C_out = w_np.shape[0]
        kernel_shape = w_np.shape[2:]

        pad_tuple = normalize_padding(
            padding, x_np.shape, kernel_shape, stride, dilation
        )

        # im2col
        cols, out_shape = im2col_nd(
            x_np, kernel_shape, stride, pad_tuple, dilation
        )

        # W: (C_out, C_in*K)
        W_col = w_np.reshape(C_out, -1)

        # out: (N, C_out, O)
        out = xp.einsum("oc,ncp->nop", W_col, cols)
        out = out.reshape((N, C_out, *out_shape))

        # ============
        # grad
        # ============
        def grad_fn(g):
            g2 = g.reshape(N, C_out, -1)  # (N, C_out, O_total)

            # -------- dW (vectorized) --------
            # cols: (N, C_in*K_total, O_total)
            # g2:   (N, C_out, O_total)
            dW = xp.einsum('nop,ncp->ocp', g2, cols).reshape(w_np.shape)

            # -------- dx --------
            # broadcast W_col.T across batch automatically via einsum
            Wb = W_col.T[None, :, :]  # (1, C_in*K, C_out)
            dCols = xp.einsum('bkc,boc->bok', Wb, g2)  # (N, C*K, O)
            
            dx = col2im_nd(dCols, x_np.shape, kernel_shape,
                            stride, pad_tuple, dilation, out_shape)

            return as_nd(dx), as_nd(dW)


        return as_nd(out), (x_np, w_np), grad_fn

    return MakeOP(_fun)(x, w)
