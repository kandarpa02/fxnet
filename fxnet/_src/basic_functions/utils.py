from ..core import primitive
import torch

@primitive
def unbroadcast(shape_like, grad):
    target = shape_like.shape
    gshape = grad.shape

    if len(gshape) > len(target):
        for _ in range(len(gshape) - len(target)):
            grad = grad.sum(axis=0)

    for i, (g, t) in enumerate(zip(grad.shape, target)):
        if t == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


def like_tensor(x):
    from ..tensor_base import Texor
    @primitive
    def ltensor(x):
        return torch.Tensor.as_subclass(x, Texor)
    
    ltensor.defvjp(
        lambda x: (torch.Tensor.as_subclass(x, Texor), []),
        lambda g, res: (g,)
    )
    return ltensor(x)