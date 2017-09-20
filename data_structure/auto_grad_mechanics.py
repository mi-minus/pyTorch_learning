# coding:utf-8
import torch
import numpy as np
from torch.autograd import Variable


def test1():
    x = Variable(torch.randn(5, 5))
    y = Variable(torch.randn(5, 5))
    z = Variable(torch.randn(5, 5), requires_grad=True)
    a = x + y
    print a.requires_grad   # False
    b = a + z
    print b.requires_grad   # True


# http://www.jianshu.com/p/cbce2dd60120  [强烈推荐]
def test_grad():
    x = Variable(torch.ones(2), requires_grad=True)
    print x.grad
    z = 4 * x * x
    y = z.norm()
    print y
    y.backward()
    print x.grad
    print torch.cuda.is_available()

if __name__ == '__main__':
    print '#### test1 ####'
    test1()
    print '#### test grad ####'
    test_grad()