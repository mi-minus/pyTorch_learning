# coding:utf-8
import torch
import numpy as np

"""
pytorch 初学习和
"""

def test1():
    """
    torch tensor 与 numpy之间的转换
    """
    np_data = np.arange(6).reshape((2,3))
    torch_data = torch.from_numpy(np_data)
    tensor2array = torch_data.numpy()
    print np_data
    print torch_data
    print tensor2array

def test2():
    """
    两者的许多函数使用方法都是一致的
    """
    data = [-1, -2, 1, 2]
    tensor = torch.FloatTensor(data)
    print "abs func:"
    print np.abs(data)
    print torch.abs(tensor)

    print "sin func:"
    print np.sin(data)
    print torch.sin(tensor)

    print "means func:"
    print np.mean(data)
    print torch.mean(tensor)

def test3():
    """
    函数矩阵相乘，保持一致
    """
    data = [[1,2],[3,4]]
    tensor = torch.FloatTensor(data)

    print "matmul func:"
    print np.matmul(data, data)
    print torch.matmul(tensor, tensor)

if __name__ == "__main__":
    test1()
    test2()
    test3()

