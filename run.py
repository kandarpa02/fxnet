import faketensor as ft 
from faketensor import ndarray as nd
import numpy as np

class Linear(ft.Cell):
    def __init__(self, _in, out):
        super().__init__()
        np.random.seed(0)
        self.w = ft.Variable(np.random.rand(_in, out), name='weight')
        self.b = ft.Variable(np.zeros(out), name='bias')


    def call(self, x):
        return ft.matmul(x, self.w) + self.b

class Model(ft.Cell):
    def __init__(self):
        super().__init__()
        self.f1 = Linear(5, 3)
        self.f2 = Linear(3, 1)

    def call(self, x):
        return self.f2(self.f1(x))
    

model = Model()

np.random.seed(0)
a = nd.array(np.random.rand(4, 5))
params = list(model.parameters())
print(params[0])

# print(ft.value_and_grad(lambda model: model(a))(model))

a = {'d':nd.array(4.), 'g':nd.array(2.)}
b = ft.Variable(4.)

def fun(dic, x, y): 
    return (dic.get('d')*x) ** dic.get('g') / y

# print(ft.grad(lambda d, y:fun(d, 4., y))(a, b))