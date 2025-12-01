import faketensor as ft 
import numpy as np
from faketensor import ndarray as nd

class Linear(ft.nn.Cell):
    def __init__(self, _in, out):
        super().__init__(name='linear')
        np.random.seed(0)
        self.weights = ft.Variable(np.random.rand(_in, out))
        self.bias = ft.Variable(np.zeros(out))


    def call(self, x):
        return ft.matmul(x, self.weights) + self.bias

class Model(ft.nn.Cell):
    def __init__(self):
        super().__init__()
        self.f1 = Linear(3, 5)
        self.f2 = Linear(5, 2)
        self.f3 = Linear(2, 1)

    def call(self, x):
        return self.f3(self.f2(self.f1(x)))
    

model = Model()
optimizer = ft.optimizers.SGD(model, lr=0.2)

np.random.seed(0)
a = nd.array(np.random.rand(20, 3))
b = nd.array(np.random.rand(20))

def loss_f(model, x, y):
    pred = model(x)
    loss = ft.mean((pred - y) ** 2)
    return loss

print("Params\n", list(model.parameters()))

for i in model.parameters():
    if i.name in ['Model.f1.weights', 'Model.f2.weights']:
        print(i.name)
        i.freeze()

print("Train Params\n", list(model.trainable_parameters()))

out, grads = ft.value_and_grad(lambda model:loss_f(model, a, b))(model)


optimizer.update(grads)

state = optimizer.get_state()

optimizer.load_state(state)

print(optimizer)
