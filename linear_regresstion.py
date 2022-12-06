"""
线性模型看作是单层的神经网络
"""
import matplotlib as plt
import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    #生成y = wx + b +噪声
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# print('features:', features[0], '\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(),
                labels.detach().numpy(), 1)
# d2l.plt.show()

#读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    #随机读取
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
                indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
#返回值和编号后，继续执行这个函数（当前值）

batch_size = 10

for x, y in data_iter(batch_size, features, labels):
    print(x , '\n', y)
    break

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(x, w, b):
    #model
    return torch.matmul(x, w) + b

def squared_loss(y_hat, y):
    #均方误差
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    #小批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_() 

lr = 0.03
num_epoch = 5
net = linreg
loss = squared_loss

for epoch in range(num_epoch):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)

    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
