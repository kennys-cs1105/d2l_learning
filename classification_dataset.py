import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib as plt

d2l.use_svg_display()

#下载dataset
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data", train=True,
    transform=trans, download=True
)
minst_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False,
    transform=trans, download=True
)
# print(len(mnist_train), len(minst_test))
# print(mnist_train[0][0].shape)
# 60000 10000
# torch.Size([1, 28, 28]

#划分dataset
def get_fashion_mnist_labels(labels):
    #返回dataset的文本标签
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress',
                    'coat', 'sandal', 'shirt', 'sneaker',
                    'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    #plot a list of imgs
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
    # d2l.plt.show()

x, y = next(iter(data.dataloader.DataLoader(mnist_train, batch_size=18)))
# x, y = next(iter(DataLoader(mnist_train, batch_size=18)))
show_images(x.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

batch_size = 256

train_iter = data.dataloader.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers=4)

timer = d2l.Timer()

for x, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')
