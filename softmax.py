"""
分类：通常为多个输出；输出i为预测第i类的置信度
对类别进行一位有效编码one-hot；使用均方损失训练；最大值为预测
需要更置信的识别正确类（正确类的置信度远远大于其他）
输出匹配概率（softmax），概率y和y_hat的区别作为损失
"""
import torch
import torch.nn as nn
from d2l import torch as d2l

batch_size = 8
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)




