from ...src.functions import mean, reshape, max
from ...src.functions.convolution import im2col_nd

# def _check(k, s):
#     assert k == s, "This pooling implementation requires kernel_size == stride"

# def _normalize_pool_param(x, dims, name):
#     if isinstance(x, int):
#         return (x,) * dims
#     if isinstance(x, (tuple, list)):
#         assert len(x) == dims, (
#             f"{name} must have {dims} values, got {len(x)}"
#         )
#         return tuple(x)
#     raise TypeError(f"Invalid {name} type: {type(x)}")


# def _check_pool_params(kernel_size, stride, dims):
#     k = _normalize_pool_param(kernel_size, dims, "kernel_size")
#     s = _normalize_pool_param(stride, dims, "stride")

#     assert k == s, (
#         "This pooling implementation requires kernel_size == stride "
#         f"(got kernel_size={k}, stride={s})"
#     )

#     # enforce uniform pooling window (reshape requirement)
#     assert len(set(k)) == 1, (
#         "Non-uniform kernel sizes are not supported by reshape-based pooling"
#     )

#     return k[0]   # scalar window size


# def max_pool1d(x, kernel_size=2, stride=2):
#     s = _check_pool_params(kernel_size, stride, dims=1)

#     N, C, L = x.shape
#     L_out = L // s

#     x = x[:, :, :L_out * s]
#     x = reshape(x, [N, C, L_out, s])

#     return max(x, axis=3)


# def avg_pool1d(x, kernel_size=2, stride=2):
#     s = _check_pool_params(kernel_size, stride, dims=1)

#     N, C, L = x.shape
#     L_out = L // s

#     x = x[:, :, :L_out * s]
#     x = reshape(x, [N, C, L_out, s])

#     return mean(x, axis=3)


# def max_pool2d(x, kernel_size=2, stride=2):
#     s = _check_pool_params(kernel_size, stride, dims=2)

#     N, C, H, W = x.shape
#     H_out = H // s
#     W_out = W // s

#     x = x[:, :, :H_out * s, :W_out * s]
#     x = reshape(x, [N, C, H_out, s, W_out, s])

#     return max(x, axis=(3, 5))


# def avg_pool2d(x, kernel_size=2, stride=2):
#     s = _check_pool_params(kernel_size, stride, dims=2)

#     N, C, H, W = x.shape
#     H_out = H // s
#     W_out = W // s

#     x = x[:, :, :H_out * s, :W_out * s]
#     x = reshape(x, [N, C, H_out, s, W_out, s])

#     return mean(x, axis=(3, 5))


# def max_pool3d(x, kernel_size=2, stride=2):
#     s = _check_pool_params(kernel_size, stride, dims=3)

#     N, C, D, H, W = x.shape
#     D_out = D // s
#     H_out = H // s
#     W_out = W // s

#     x = x[:, :, :D_out * s, :H_out * s, :W_out * s]
#     x = reshape(
#         x,
#         [N, C,
#          D_out, s,
#          H_out, s,
#          W_out, s]
#     )

#     return max(x, axis=(3, 5, 7))


# def avg_pool3d(x, kernel_size=2, stride=2):
#     s = _check_pool_params(kernel_size, stride, dims=3)

#     N, C, D, H, W = x.shape
#     D_out = D // s
#     H_out = H // s
#     W_out = W // s

#     x = x[:, :, :D_out * s, :H_out * s, :W_out * s]
#     x = reshape(
#         x,
#         [N, C,
#          D_out, s,
#          H_out, s,
#          W_out, s]
#     )

#     return mean(x, axis=(3, 5, 7))

def _can_use_reshape_pool(k, s, p, d): # UNUSED FOR NOW
    return (
        k == s
        and len(set(k)) == 1
        and all(pi == 0 for pi in p)
        and all(di == 1 for di in d)
    )


def _normalize_nd(x, dims, name):
    if isinstance(x, int):
        return (x,) * dims
    if isinstance(x, (tuple, list)):
        assert len(x) == dims, f"{name} must have {dims} values"
        return tuple(x)
    raise TypeError(f"Invalid {name}: {type(x)}")

def _pool_nd_im2col(
    x,
    kernel_size,
    stride,
    padding,
    dilation,
    reduce,      # max or mean
):
    dims = x.ndim - 2

    k = _normalize_nd(kernel_size, dims, "kernel_size")
    s = _normalize_nd(stride, dims, "stride")
    p = _normalize_nd(padding, dims, "padding")
    d = _normalize_nd(dilation, dims, "dilation")

    cols, out_shape = im2col_nd(
        x,
        kernel_shape=k,
        stride=s,
        padding=p,
        dilation=d,
    )
    # cols shape: (N, C * prod(k), prod(out_shape))

    N, CK, O = cols.shape
    C = x.shape[1]
    K_total = CK // C

    cols = reshape(cols, [N, C, K_total, O])

    y = reduce(cols, axis=2)  # pool over kernel

    return reshape(y, [N, C, *out_shape])

def max_pool1d(x, kernel_size=2, stride=2, padding=0, dilation=1):
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=max,
    )


def avg_pool1d(x, kernel_size=2, stride=2, padding=0, dilation=1):
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=mean,
    )

def max_pool2d(x, kernel_size=2, stride=2, padding=0, dilation=1):
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=max,
    )


def avg_pool2d(x, kernel_size=2, stride=2, padding=0, dilation=1):
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=mean,
    )

def max_pool3d(x, kernel_size=2, stride=2, padding=0, dilation=1):
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=max,
    )


def avg_pool3d(x, kernel_size=2, stride=2, padding=0, dilation=1):
    return _pool_nd_im2col(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        reduce=mean,
    )
