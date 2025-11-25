from faketensor.src.jit.placeholder import FT_Tracer
from faketensor.src.jit.executor import FT_Function
import faketensor as ft 
from faketensor import ndarray as nd

# a = FT_Tracer((), 'float32', 'a')
# b = FT_Tracer((), 'float32', 'b')
# c = FT_Tracer((), 'float32', 'c')
# out = (a * b)+a

# f = FT_Function(out, [a, b])

# print('func:\n', f)
# e = f.compile()

# print('out:\n',e(2., 3.))

a = nd.array(4.)
b = nd.array(5.)
g = ft.grad(lambda x, y:x/y)
# print(g(a, b))

name = None
current_id = 0

def gen_name():
    global name
    global current_id
    import string
    import random
    depth = 10000000
    current_alp = random.choice(string.ascii_letters)
    
    name = f'{current_alp}{current_id}'
    _c = 1 + int(name[-1])
    current_id = _c if not _c>depth else 0

for i in range(10):
    gen_name()
    print(name)