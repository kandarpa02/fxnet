from .basic_functions.vjps import add, mul

def norm_tracer(a):
    if isinstance(a, Tracer):
        return a.value
    return a

class Tracer:
    def __init__(self, value, node=None):
        self.value = value
        self.node = node

    def __repr__(self):
        return repr(self.value)
    
    def __add__(self, other): return add(self, other)
    def __radd__(self, other): return add(other, self)
    def __mul__(self, other): return mul(self, other)
    def __rmul__(self, other): return mul(other, self)


