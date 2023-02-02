"""
vgg:组成卷积块
更多的3*3比5*5更有效，增加网络深度
更大更深的alexnet，重复的vgg块
"""
import torch
import torch.nn as nn
from d2l import torch as d2l

def vgg_block(num_convs, in_ch, out_ch):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_ch = out_ch
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

#层数+通道数
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_ch = 1
    for (num_convs, out_ch) in conv_arch:
        conv_blks.append(vgg_block(
            num_convs, in_ch, out_ch))
        
        in_ch = out_ch
    # return nn.Sequential(*conv_blks, nn.Flatten())

    return nn.Sequential(*conv_blks, nn.Flatten(),
    nn.Linear(out_ch*7*7, 4096), nn.ReLU(),
    nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(0.5), nn.Linear(4096, 10))

# net = vgg(conv_arch)

# X = torch.randn(size=(1, 1, 224, 224))
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__, 'out shape: \t', X.shape)
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 10, 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())