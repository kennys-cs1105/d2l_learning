import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])
# print(x + y, x * y, x / y, x ** y)

# A = torch.arange(20).reshape(5, 4)
# print(A.T)

X = torch.arange(24).reshape(2, 3, 4)
# print(X)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
# print(A * B)

# a = 2
# print(a + X, (a * X).shape)

#L2范数（元素平方和的平方根）
u = torch.tensor([3.0, -4.0])
# print(torch.norm(u))

#L1范数(元素绝对值之和)
# print(torch.abs(u).sum())

#矩阵的Frobenius范数（矩阵元素的平方和的平方根）
# print(torch.norm(torch.ones(4, 9)))

"""
shapep[2, 5, 4]
按axis[1, 2]求和，留下[2]的结果，去掉axisp[1, 2]

keepdims=True
shape[2, 5, 4], axis[1], 结果为[2, 1, 4]
"""
a = torch.ones((2, 5, 4))
"""
print(a)
tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])
"""
print(a.sum(axis=1, keepdim=True).shape)
