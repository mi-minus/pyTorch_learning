#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/8/18 12:35 AM
# @Author  : minus
# @Site    : 
# @File    : classification.py
# @Software: PyCharm

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)

x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1),0).type(torch.FloatTensor)
y = torch.cat((y0, y1),).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        x = self.out(x)             # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

if __name__ == '__main__':
    net = Net(n_feature=2, n_hidden=10, n_output=2)

    print(net)

    plt.ion()       # 打开交互模式
    plt.show()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    # 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
    loss_func = torch.nn.CrossEntropyLoss()   # 用于分类

    for t in range(500):
        out = net(x)

        loss = loss_func(out, y)    # 计算两者的误差

        optimizer.zero_grad()       # 清空上一步的残余更新参数值
        loss.backward()             # 误差反向传播, 计算参数更新值
        optimizer.step()            # 将参数更新值施加到 net 的 parameters 上
        if t%2 == 0:
            plt.cla()
            # 过了一道 softmax 的激励函数后的最大概率才是预测值
            prediction = torch.max(F.softmax(out), 1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
            accuracy = sum(pred_y == target_y)/200
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size':20, 'color':'red'})
            plt.pause(0.3)


    plt.ioff()      # 显示前关掉交互模式
    plt.show()
