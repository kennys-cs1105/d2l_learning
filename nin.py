"""
网络中的网络nin
一个卷积层后跟两个全连接层：conv + 1*1conv + 1*1conv
stride=1，无填充，输出形状和卷积层输出一样

nin:无全连接层，交替使用nin块和最大池化层（stride=2）
最后使用全局平均池化得到输出（输入通道为类别数,得到概率）
"""

from matplotlib.cbook import flatten
import torch
import torch.nn as nn
from d2l import torch as d2l

def nin_block(in_ch, out_ch, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
    nn.ReLU(), nn.Conv2d(out_ch, out_ch, kernel_size=1),
    nn.ReLU(), nn.Conv2d(out_ch, out_ch, kernel_size=1),
    nn.ReLU())

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)
# X = torch.rand(size=(1, 1, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'out shape: \t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
