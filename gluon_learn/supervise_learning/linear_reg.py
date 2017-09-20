#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/20/17 11:47 PM
# @Author  : minus
# @Site    : 
# @File    : linear_reg.py
# @Software: PyCharm
import random

from mxnet import ndarray as nd
from mxnet import autograd

num_inputs = 2
num_examples = 1000
batch_size = 10

def create_dataset():
    true_w = [2, -3, 4]
    true_b = 4.2

    X = nd.random_normal(shape=(num_examples, num_inputs))
    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
    y += .01 * nd.random_normal(shape=y.shape)
    return X, y

def read_data():
    X, y = create_dataset()
    dataiter = data_iter(X, y)
    return dataiter
    # for data, label in dataiter:
    #     print data, label
    #     break

def data_iter(X, y):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i+batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y, j)

def random_init_param():
    w = nd.random_normal(shape=(num_inputs, 1))
    b = nd.zeros((1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params


def net(X):
    return nd.dot(X, w) + b


def square_loss(yhat, y):
    return (yhat-y.reshape(yhat.shape)) ** 2


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def train():
    epochs = 5
    learning_rate = 0.001
    for e in range(epochs):
        total_loss = 0
        for data, label in read_data():
            with autograd.record():
                output = net(data)


def main():
    X, y = create_dataset()
    params = random_init_param()
    nd.dot(X, params[0]) + params[1]


if __name__ == "__main__":
    # read_data()
    random_init_param()