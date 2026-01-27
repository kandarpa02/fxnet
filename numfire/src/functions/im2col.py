from ...backend import backend as b
from ..base import MakeOP
from .._typing import Array
from .xpy_utils import get_dev, module

def _to_tuple(v, dims):
    if isinstance(v, int):
        return (v,) * dims
    return tuple(v)

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


def im2col(x:Array, kernel_shape, stride, padding, dilation):
	d = get_dev(x)
	mod = module(d)
	pad = mod.pad
	arange = mod.arange
	stack = mod.stack
	meshgrid = mod.meshgrid
	broadcast_to = mod.broadcast_to

	dims = len(kernel_shape)
	kernel_shape = tuple(kernel_shape)
	stride = tuple(stride)
	dilation = tuple(dilation)
	padding = tuple(padding)
		
	def fun(_x):
		x =  getattr(_x, "__backend_buffer__", _x)
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

		def grad_fn(g):
			return col2im( 
				g,
                x_shape=x.shape,
				kernel_shape=kernel_shape,
				stride=stride,
				padding=padding,
				dilation=dilation,
                out_shape= out_shape
			),

		return (cols, out_shape), grad_fn
	
	return MakeOP(fun)(x)


from ..base import MakeOP
from .xpy_utils import get_dev, module


def col2im(
    cols,
    x_shape,
    kernel_shape,
    stride,
    padding,
    dilation,
    out_shape,
):
    d = get_dev(cols)
    mod = module(d)

    zeros = mod.zeros
    arange = mod.arange
    stack = mod.stack
    meshgrid = mod.meshgrid
    broadcast_to = mod.broadcast_to
    add = mod.add

    kernel_shape = tuple(kernel_shape)
    stride = tuple(stride)
    dilation = tuple(dilation)
    padding = tuple(padding)

    dims = len(kernel_shape)

    def fun(_cols):
        cols = getattr(_cols, "__backend_buffer__", _cols)

        N, C = x_shape[:2]
        spatial = x_shape[2:]

        padded_shape = (
            N,
            C,
            *[spatial[i] + 2 * padding[i] for i in range(dims)],
        )
        xpad = zeros(padded_shape, dtype=cols.dtype)

        # reshape cols -> (N, C, K, O)
        K = 1
        for k in kernel_shape:
            K *= k
        O = 1
        for o in out_shape:
            O *= o

        cols_rs = cols.reshape(N, C, K, O)

        # kernel offsets
        k_list = [arange(kernel_shape[i]) * dilation[i] for i in range(dims)]
        k_grid = stack(meshgrid(*k_list, indexing="ij"), axis=0)
        k_grid = k_grid.reshape(dims, K, 1)

        # output offsets
        o_list = [arange(out_shape[i]) * stride[i] for i in range(dims)]
        o_grid = stack(meshgrid(*o_list, indexing="ij"), axis=0)
        o_grid = o_grid.reshape(dims, 1, O)

        idx = k_grid + o_grid  # (dims, K, O)

        # broadcast batch/channel indices
        N_idx = arange(N).reshape(N, 1, 1, 1)
        C_idx = arange(C).reshape(1, C, 1, 1)

        N_idx = broadcast_to(N_idx, (N, C, K, O))
        C_idx = broadcast_to(C_idx, (N, C, K, O))

        full_idx = [N_idx, C_idx]
        for i in range(dims):
            full_idx.append(
                broadcast_to(
                    idx[i][None, None, :, :],
                    (N, C, K, O),
                )
            )

        # SCATTER-ADD (this is the key)
        add.at(xpad, tuple(full_idx), cols_rs)

        # crop padding
        slices = [slice(None), slice(None)]
        for i in range(dims):
            slices.append(slice(padding[i], padding[i] + spatial[i]))

        dx = xpad[tuple(slices)]

        # -----------------------
        # backward = im2col
        # -----------------------
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

        return dx, grad_fn

    return MakeOP(fun)(cols)
