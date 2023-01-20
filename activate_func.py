# from d2l import torch as d2l
# import torch
# import matplotlib as plt

# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = torch.relu(x)
# # d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
# # d2l.plt.show()
# y.backward(torch.ones_like(x), retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
# d2l.plt.show()
# -*- coding:utf-8 -*-
# from matplotlib import pyplot as plt
# import numpy as np
# import mpl_toolkits.axisartist as axisartist
# from matplotlib.pyplot import MultipleLocator

# def sigmoid(x):
#     return 1. / (1 + np.exp(-x))


# def tanh(x):
#     return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# def relu(x):
#     return np.where(x < 0, 0, x)

# def prelu(x):
#     return np.where(x < 0, 0.5 * x, x)

# def sigmoid(x):
#     #直接返回sigmoid函数
#     return 1. / (1. + np.exp(-x))
# def softmax(x):
#     return np.exp(x)/np.sum(np.exp(x), axis=0)

# def plot_sigmoid():

#     # param:起点，终点，间距
#     x = np.arange(-10, 10, 0.5)
#     print(x)
#     y = sigmoid(x)
#     #plt.plot(x, label='sigmoid')  # 添加label设置图例名称
#     #plt.title("sigmoid",fontsize=20)
#     plt.grid()
#     plt.plot(x, y,label='sigmoid',color='r')
#     plt.legend(fontsize=20)
#     plt.xlabel("x",fontsize=20)
#     plt.ylabel("f(x)",fontsize=20)
#     # 设置刻度字体大小
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.xlim([-10, 10])
#     plt.ylim([0, 1])
#     x_major_locator = MultipleLocator(5)
#     # 把x轴的刻度间隔设置为1，并存在变量里
#     y_major_locator = MultipleLocator(0.2)
#     # 把y轴的刻度间隔设置为10，并存在变量里
#     ax = plt.gca()
#     # ax为两条坐标轴的实例
#     ax.xaxis.set_major_locator(x_major_locator)
#     # 把x轴的主刻度设置为1的倍数
#     ax.yaxis.set_major_locator(y_major_locator)
#     # 把y轴的主刻度设置为10的倍数
#     plt.show()

# def plot_tanh():
#     x = np.arange(-10, 10, 0.1)
#     y = tanh(x)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.spines['top'].set_color('none')
#     ax.spines['right'].set_color('none')
#     # ax.spines['bottom'].set_color('none')
#     # ax.spines['left'].set_color('none')
#     ax.spines['left'].set_position(('data', 0))
#     ax.spines['bottom'].set_position(('data', 0))
#     ax.plot(x, y)
#     plt.xlim([-10.05, 10.05])
#     plt.ylim([-1.02, 1.02])
#     ax.set_yticks([-1.0, -0.5, 0.5, 1.0])
#     ax.set_xticks([-10, -5, 5, 10])
#     plt.tight_layout()
#     plt.savefig("tanh.png")
#     plt.show()

# def plot_relu():
#     x = np.arange(-10, 10, 0.1)
#     y = relu(x)
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # ax.spines['top'].set_color('none')
#     # ax.spines['right'].set_color('none')
#     # # ax.spines['bottom'].set_color('none')
#     # # ax.spines['left'].set_color('none')
#     # ax.spines['left'].set_position(('data', 0))
#     # ax.plot(x, y)
#     # plt.xlim([-10.05, 10.05])
#     # plt.ylim([0, 10.02])
#     # ax.set_yticks([2, 4, 6, 8, 10])
#     # plt.tight_layout()
#     # plt.savefig("relu.png")
#     #plt.title("relu")
#     plt.grid()
#     plt.plot(x, y, label='relu', color='r')
#     plt.legend(fontsize=20)
#     plt.xlabel("x",fontsize=20)
#     plt.ylabel("f(x)",fontsize=20)
#     # 设置刻度字体大小
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.xlim([-10, 10])
#     plt.ylim([0, 10])
#     x_major_locator = MultipleLocator(5)
#     # 把x轴的刻度间隔设置为1，并存在变量里
#     y_major_locator = MultipleLocator(5)
#     # 把y轴的刻度间隔设置为10，并存在变量里
#     ax = plt.gca()
#     # ax为两条坐标轴的实例
#     ax.xaxis.set_major_locator(x_major_locator)
#     # 把x轴的主刻度设置为1的倍数
#     ax.yaxis.set_major_locator(y_major_locator)
#     # 把y轴的主刻度设置为10的倍数
#     plt.show()

# def plot_prelu():
#     x = np.arange(-10, 10, 0.1)
#     y = prelu(x)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.spines['top'].set_color('none')
#     ax.spines['right'].set_color('none')
#     # ax.spines['bottom'].set_color('none')
#     # ax.spines['left'].set_color('none')
#     ax.spines['left'].set_position(('data', 0))
#     ax.spines['bottom'].set_position(('data', 0))
#     ax.plot(x, y)
#     plt.xticks([])
#     plt.yticks([])
#     plt.tight_layout()
#     plt.savefig("prelu.png")
#     plt.show()

# def plot_softmax():
#     x = np.arange(-10, 10, 0.1)
#     y = softmax(x)
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # ax.spines['top'].set_color('none')
#     # ax.spines['right'].set_color('none')
#     # # ax.spines['bottom'].set_color('none')
#     # # ax.spines['left'].set_color('none')
#     # ax.spines['left'].set_position(('data', 0))
#     # ax.plot(x, y)
#     # plt.xlim([-10.05, 10.05])
#     # plt.ylim([0, 10.02])
#     # ax.set_yticks([2, 4, 6, 8, 10])
#     # plt.tight_layout()
#     # plt.savefig("relu.png")
#     #plt.title("softmax")
#     plt.grid()
#     plt.plot(x, y, label='softmax', color='r')
#     plt.legend(fontsize=20)
#     plt.xlabel("x",fontsize=20)
#     plt.ylabel("f(x)",fontsize=20)
#     plt.xlim([-10, 10])
#     plt.ylim([0, 0.1])
#     # 设置刻度字体大小
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     x_major_locator = MultipleLocator(5)
#     # 把x轴的刻度间隔设置为1，并存在变量里
#     y_major_locator = MultipleLocator(0.02)
#     # 把y轴的刻度间隔设置为10，并存在变量里
#     ax = plt.gca()
#     # ax为两条坐标轴的实例
#     ax.xaxis.set_major_locator(x_major_locator)
#     # 把x轴的主刻度设置为1的倍数
#     ax.yaxis.set_major_locator(y_major_locator)
#     # 把y轴的主刻度设置为10的倍数
#     plt.show()
# if __name__ == "__main__":
#     #plot_sigmoid()
#     #plot_tanh()
#     #plot_relu()
#     plot_softmax()
#     #plot_prelu()
import math
import numpy as np
import matplotlib.pyplot as plt

# set x's range
x = np.arange(-10, 10, 0.1)

y1 = 1 / (1 + math.e ** (-x))  # sigmoid
# y11=math.e**(-x)/((1+math.e**(-x))**2)
y11 = 1 / (2 + math.e ** (-x)+ math.e ** (x))  # sigmoid的导数

y2 = (math.e ** (x) - math.e ** (-x)) / (math.e ** (x) + math.e ** (-x))  # tanh
y22 = 1-y2*y2  # tanh函数的导数

y3 = np.where(x < 0, 0, x)  # relu
y33 = np.where(x < 0, 0, 1)  # ReLU函数导数

plt.xlim(-4, 4)
plt.ylim(-1, 1.2)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# Draw pic
# plt.plot(x, y1, label='Sigmoid', linestyle="-", color="red")
# plt.plot(x, y11, label='Sigmoid derivative', linestyle="-", color="violet")
# plt.legend(['Sigmoid', 'Sigmoid derivative'])  # y1 y11
# plt.show()

# plt.plot(x, y2, label='Relu', linestyle="-", color="blue")
# plt.plot(x, y22, label='Relu derivative', linestyle="-", color="violet")
# plt.legend(['Relu', 'Relu derivative'])  # y2 y22
# plt.show()

plt.plot(x, y3, label='Tanh', linestyle="-", color="green")
plt.plot(x, y33, label='Tanh derivative', linestyle="-", color="violet")
plt.legend(['Tanh', 'Tanh derivative'])  # y3 y33
plt.show()

# # Title
# plt.legend(['Sigmoid', 'Tanh', 'Relu'])
# plt.legend(['Sigmoid', 'Sigmoid derivative'])  # y1 y11
# plt.legend(['Relu', 'Relu derivative'])  # y2 y22
# plt.legend(['Tanh', 'Tanh derivative'])  # y3 y33

# plt.legend(['Sigmoid', 'Sigmoid derivative', 'Relu', 'Relu derivative', 'Tanh', 'Tanh derivative'])  # y3 y33
# # plt.legend(loc='upper left')  # 将图例放在左上角

# # save pic
# # plt.savefig('plot_test.png', dpi=100)
# plt.savefig(r"./SRL_result")

# # show it!!
# plt.show()

