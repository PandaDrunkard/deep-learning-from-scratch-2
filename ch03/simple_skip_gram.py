import sys, os
sys.path.append(os.pardir)

from common.layer import MatMul, SoftmaxWithLoss

import numpy as np

class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer0 = SoftmaxWithLoss()
        self.loss_layer1 = SoftmaxWithLoss()

        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)

        l1 = self.loss_layer0.forward(s, contexts[:, 0])
        l2 = self.loss_layer1.forward(s, contexts[:, 1])
        loss = l1 + l2

        return loss
    
    def backward(self, dout=1):
        dl0 = self.loss_layer1.backward(dout)
        dl1 = self.loss_layer0.backward(dout)
        ds = dl0 + dl1
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None
        