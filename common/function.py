# coding: utf-8
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    if x.ndim == 2:
        return softmax_2D(x)
    else:
        return softmax_1D(x)

def softmax_1D(x):
    c = np.max(x)
    exp_a = np.exp(x - c) # avoid overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def softmax_2D(x):
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    delta = 1e-7

    # yの各ベクトルについて、正解ラベルのインデックスにある要素を取り出す
    biggests_in_y = y[np.arange(batch_size), t]

    return -1 * np.sum(np.log(biggests_in_y + delta)) / batch_size