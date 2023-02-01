import torch
import torch.nn as nn

def comp_conv2d(conv2d, x):
    x = x.reshape((1, 1) + x.shape)#增加通道数和批次大小
    y = conv2d(x)
    return y.reshape(y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
x = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, x).shape)