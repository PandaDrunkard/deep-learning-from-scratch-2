from layer import Convolution, Pooling, Affine
import numpy as np


def affine_test():
    print("========== Affine ==========")
    x = np.random.randn(99,30,12,12)
    W = np.random.randn(30*12*12, 100)
    b = np.random.randn(100)

    layer = Affine(W, b)

    dout = layer.forward(x)
    dx = layer.backward(dout)

    print(dout.shape)
    print(dx.shape)
    
def convolution_test():
    print("========== Convolution ==========")
    x = np.random.randn(99, 1, 28, 28)
    W = np.random.randn(30, 1, 5, 5)
    b = np.zeros(30)

    layer = Convolution(W, b, stride=1, pad=0)

    dout = layer.forward(x)
    dx = layer.backward(dout)

    print(dout.shape)
    print(dx.shape)

def pooling_test():
    print("========== Pooling ==========")
    x = np.random.randn(99,30,24,24)
    
    layer = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)

    dout = layer.forward(x)
    dx = layer.backward(dout)

    print(dout.shape)
    print(dx.shape)

affine_test()
convolution_test()
pooling_test()