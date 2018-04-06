# coding:utf-8
import mxnet

from mxnet import ndarray as nd
from mxnet import autograd as ag
import numpy as np

def test1():
    x = nd.zeros((3,4))
    print(x)
    print(nd.ones((4,4)))
    print(nd.array([[1,2,3],[4,5,6]]))
    tmp1 = nd.random_normal(0, 1, shape=(3, 4))
    print(tmp1)
    print(tmp1.shape)
    print(tmp1.size)

    print(x + tmp1)

    print(nd.exp(tmp1))

def test2():
    x = np.ones((3,4))
    print(x)
    y = nd.array(x)  # numpy to mxnet.ndarray
    print(y)
    print(y.asnumpy())

if __name__ == "__main__":
    test1()
    test2()

