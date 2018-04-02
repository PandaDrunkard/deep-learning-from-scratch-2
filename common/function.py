# coding: utf-8
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    if x.ndim == 2:
        c = x.max(axis=1, keepdims=True)
        x = np.exp(x - c)
        x /= x.sum(axis=1, keepdims=True)
    else:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

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