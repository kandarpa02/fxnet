from .dataset_base import dataset, cache_path

__all__ = ["MNIST"]

class MNIST(dataset):
    def __init__(self, path=None):
        if path is None:
            path = cache_path("mnist.npz")
            
        super().__init__(
            path='mnist.npz' if path==None else path, 
            link="https://drive.google.com/file/d/1Bmft6ApTEqadnWR5m-7Ozuy7x3pAnlYx/view?usp=sharing"
            )
    