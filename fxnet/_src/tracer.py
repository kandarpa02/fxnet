class Tracer:
    def __init__(self, value, node=None):
        self.value = value
        self.node = node

    # def __add__(self, other):
    #     return add(self, other)
    # def __radd__(self, other):
    #     return add(self, other)
    # def __mul__(self, other):
    #     return mul(self, other)
    # def __rmul__(self, other):
    #     return mul(self, other)

    def __repr__(self):
        return f"Tracer({self.value})"
    