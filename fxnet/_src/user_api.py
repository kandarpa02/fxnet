from .differentiate_engine import grad, value_and_grad
from .tensor_base import Tracer 
import torch

def tensor(x):
    if isinstance(x, torch.Tensor):
        return Tracer(x)
    else:
        return Tracer(torch.tensor(x))