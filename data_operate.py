import torch

# x = torch.arange(12)
# # print(x.shape)
# X = x.reshape(3, 4)

x = torch.tensor([1, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
# print(x + y, x * y, x / y, x ** y)
# print(x == y)
"""
也可以写成：
x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((x, y), dim=0), torch.cat((x, y), dim=1)
"""

"""
广播机制，形状不一样，维度需要保持一致
把a复制成3*2的矩阵，把b也复制成3*2的矩阵  
"""
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)
"""
tensor([[0],
        [1],
        [2]]) 
tensor([[0, 1]])
"""
print(a + b)
"""
tensor([[0, 1],
        [1, 2],
        [2, 3]])
"""

"""
id:相当于cpp的指针，避免内存消耗
"""
before = id(y)
y = y + x
print(id(y) == before)
