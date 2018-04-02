import sys, os
sys.path.append(os.pardir)
import numpy as np
from util import im2col, col2im

def im2col_test():
    x = np.random.randn(5, 1, 28, 28)
    col = im2col(x, 4, 4, stride=1, pad=0)

    print(x.shape)
    print(col.shape)

def col2im_test():
    col = np.random.randn(3125, 16)
    x = col2im(col, (5, 1, 28, 28), 4, 4, stride=1, pad=0)

    print(col.shape)
    print(x.shape)

im2col_test()
col2im_test()