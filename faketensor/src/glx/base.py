class Proxy:
    def repr(self):
        pass

    def __repr__(self):
        return self.repr()

class DTYPE(Proxy):
    def __init__(self):
        pass
    def repr(self):
        return f'ProxyDTYPE'

class SHAPE(Proxy):
    def __init__(self, *shape):
        self.shape = shape

    def repr(self):
        return f"ProxySHAPE({list(self.shape)})"

# class FLOAT32(DTYPE):
#     def __init__(self):
#         super().__init__()
#         self.dt = np.float32
    
#     def repr(self):
#         return f'ProxyFLOAT32'

class ProxyArray(Proxy):
    def __init__(self, name:str, dtype:type[DTYPE], shape:SHAPE=SHAPE()):
        self.name = name
        self.dtype = dtype()
        self.shape = shape

    def repr(self):
        return self.name
    
class ProxyParams(ProxyArray):
    def __init__(self, name: str, dtype: type[DTYPE], shape: SHAPE = SHAPE()):
        super().__init__(name, dtype, shape)
