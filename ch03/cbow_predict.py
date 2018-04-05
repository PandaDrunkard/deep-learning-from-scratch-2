import sys, os
sys.path.append(os.pardir)
from common.layer import MatMul

import numpy as np

c0 = np.zeros(7)
c1 = np.zeros(7)
c0[0] = 1
c1[2] = 1

W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = (h0 + h1) / 2.0
s = out_layer.forward(h)

print(s)