"""
更多的层不一定提高精度
"""

from turtle import forward
from unittest import result
import torch
import torch.nn as nn
from torch.nn import functional as F
from d2l import torch as d2l

class residual(nn.Module):
    def __init__(self, in_ch, num_ch, use_1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, num_ch, kernel_size=3,
            padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_ch, num_ch, kernel_size=3, padding=1)

        if use_1conv:
            self.conv3 = nn.Conv2d(in_ch, num_ch, kernel_size=1,
                stride=strides)
        else:
            self.conv3 = None
        
        self.bn1 = nn.BatchNorm2d(num_ch)
        self.bn2 = nn.BatchNorm2d(num_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)

# blk = residual(3, 3)
# x = torch.rand(4, 3, 6, 6)
# Y = blk(x)
# print(Y.shape)
 
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(in_ch, num_ch, num_residual, first_block=False):
    blk = []
    for i in range (num_residual):
        if i == 0 and not first_block:
            blk.append(residual(in_ch, num_ch,
                use_1conv=True, strides=2))
        else:
            blk.append(residual(num_ch, num_ch))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5, 
    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)