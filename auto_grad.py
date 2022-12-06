import torch

#假设对y=2x.Tx关于x求导
x = torch.arange(4.0, requires_grad=True)#存梯度

# y = 2 * torch.dot(x, x)
# y.backward()
# print(x.grad )

"""
默认情况下，pytorch会累计梯度，需要清除之前的值
"""
# x.grad.zero_()
# y = x.sum()
# y.backward()
# print(x.grad)
