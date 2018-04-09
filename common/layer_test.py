import unittest

import sys, os
sys.path.append(os.pardir)

from common.layer import MatMul, Embedding
from common.np import *

class LayerTest(unittest.TestCase):
    def test_matmul(self):
        W = np.random.randn(7, 3)
        x = np.random.randn(10, 7)

        matmul = MatMul(W)
        dout = matmul.forward(x)
        dx = matmul.backward(dout)

        np.testing.assert_array_almost_equal(dout.shape, (10, 3))
        np.testing.assert_array_almost_equal(dx.shape, (10, 7))

    def test_embedding(self):
        W = np.random.randn(7, 3)
        idx = 1

        embedding = Embedding(W)
        dout = embedding.forward(idx)
        embedding.backward(dout)

        np.testing.assert_array_almost_equal(dout.shape, (3,))
        np.testing.assert_array_almost_equal(embedding.grads[0][idx], dout)

if __name__ == '__main__':
    unittest.main()