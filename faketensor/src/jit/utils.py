import numpy as np

name = None
current_id = 0

def next_name():
    global name
    global current_id
    import string
    import random
    depth = 10000000
    current_alp = random.choice(string.ascii_letters)
    
    name = f'{current_alp}{current_id}'
    _c = 1 + int(name[-1])
    current_id = _c if not _c>depth else 0

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
    
