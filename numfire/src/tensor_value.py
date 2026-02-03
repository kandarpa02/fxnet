import torch
import numfire.src.primitives as p
from .DType import normalize_dtype, DType
from ._typing import TensorLike

class TensorBox(torch.Tensor):
    def __new__(cls, data, node=None, trace_id=None, dtype=None):
        default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if isinstance(data, torch.Tensor|TensorBox):
            default_device = data.device
        t = torch.as_tensor(data).to(dtype).detach()
        obj = torch.Tensor._make_subclass(cls, t, require_grad=False)
        setattr(obj, 'node', node)
        setattr(obj, 'trace_id', trace_id)
        return obj

    def __repr__(self):
        # _dstr = DType.from_torch_dtype(str(self.dtype)).name
        return "Tensor" + super().__repr__().removeprefix("TensorBox").replace('torch.', 'numfire.')
    
    __str__ = __repr__

wrap = lambda args: tuple(TensorBox(arg) for arg in args)

defmethod = lambda name, func: setattr(TensorBox, name, func)


defmethod('__add__',  lambda x, y: p.add(x, y))
defmethod('__radd__', lambda x, y: p.add(y, x))

defmethod('__sub__',  lambda x, y: p.sub(x, y))
defmethod('__rsub__', lambda x, y: p.sub(y, x))

defmethod('__mul__',  lambda x, y: p.mul(x, y))
defmethod('__rmul__', lambda x, y: p.mul(y, x))

defmethod('__truediv__',  lambda x, y: p.div(x, y))
defmethod('__rtruediv__', lambda x, y: p.div(y, x))

defmethod('__pow__',  lambda x, y: p.power(x, y))
defmethod('__rpow__', lambda x, y: p.power(y, x))


# -------- unary --------
defmethod('__neg__', lambda x: p.neg(x))
defmethod('__pos__', lambda x: x)


# -------- comparisons --------
defmethod('__lt__', lambda x, y: p.less(x, y))
defmethod('__le__', lambda x, y: p.less_equal(x, y))
defmethod('__gt__', lambda x, y: p.greater(x, y))
defmethod('__ge__', lambda x, y: p.greater_equal(x, y))
defmethod('__eq__', lambda x, y: p.equal(x, y))
defmethod('__ne__', lambda x, y: p.not_equal(x, y))


# -------- logical --------
defmethod('__and__', lambda x, y: p.logical_and(x, y))
defmethod('__rand__', lambda x, y: p.logical_and(y, x))

defmethod('__or__', lambda x, y: p.logical_or(x, y))
defmethod('__ror__', lambda x, y: p.logical_or(y, x))

defmethod('__xor__', lambda x, y: p.logical_xor(x, y))
defmethod('__rxor__', lambda x, y: p.logical_xor(y, x))

defmethod('__invert__', lambda x: p.logical_not(x))


# -------- matrix --------
defmethod('__matmul__',  lambda x, y: p.matmul(x, y))
defmethod('__rmatmul__', lambda x, y: p.matmul(y, x))


DtypeLike = DType|str|None

class Variable(TensorBox):
    def __new__(
        cls,
        data:TensorLike,
        dtype:DtypeLike=None,
        trainable: bool = True,
        node=None,
        trace_id=None,
        name: str | None = None
    ):
        obj = super().__new__(
            cls,
            data,
            node=node,
            trace_id=trace_id,
            dtype=dtype,
        )

        object.__setattr__(obj, "id", name)
        object.__setattr__(obj, "_trainable", trainable)
        return obj

    # TensorFlow-style in-place update
    def assign(self, value):
        with torch.no_grad():
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
            self.copy_(value)

    def __repr__(self):
        # _dstr = DType.from_torch_dtype(str(self.dtype)).name
        base = "Tensor" + super().__repr__().removeprefix("TensorVariable")
        return (
            f"Variable(name={self.id!r}, "
            f"shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, "
            f"trainable={self._trainable})\n"
            f"{base}"
        ).replace('torch.', 'numfire.')
        # return base

    __str__ = __repr__

    name = property(lambda self: self.id)
    trainable = property(lambda self: self._trainable)

