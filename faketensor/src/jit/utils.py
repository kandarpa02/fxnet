import numpy as np


class Meta:
    @staticmethod
    def element_wise_shape(a, b):
        a_shape, b_shape = a.shape, b.shape
        dummy1 = np.empty(a_shape)
        dummy2 = np.empty(b_shape)
        return (dummy1 + dummy2).shape
    
    @staticmethod
    def dot_shape(a, b):
        a_shape, b_shape = a.shape, b.shape
        dummy1 = np.empty(a_shape)
        dummy2 = np.empty(b_shape)
        return (dummy1 @ dummy2).shape

    @staticmethod
    def DType(a, b):
        dummy1 = np.empty(a.shape, a.dtype)
        dummy2 = np.empty(b.shape, b.dtype)
        return (dummy1 + dummy2).dtype.__str__()