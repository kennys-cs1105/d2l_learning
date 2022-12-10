import torch
from d2l import torch as d2l
import torch.nn as nn
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

w1 = torch.randn(num_inputs, num_hiddens, requires_grad=True)
b1 = torch.zeros(num_hiddens, requires_grad=True)
w2 = torch.randn(num_hiddens, num_outputs, requires_grad=True)
b2 = torch.zeros(num_outputs, requires_grad=True)

params = [w1, b1, w2, b2]

def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)

#model
def net(x):
    x = x.reshape((-1, num_inputs))
    h = relu(x @ w1 + b1)
    return (h @ w2 + b2)

loss=nn.CrossEntropyLoss(reduction='none')

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

"""
报错：OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized
1.
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
2.
env中查找libiomp5md.dll，挨个剪切，看能不能跑
"""
