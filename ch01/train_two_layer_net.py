import numpy as np

import sys, os
sys.path.append(os.pardir)
from common.trainer import Trainer
from common.optimizer import SGD
from dataset import spiral
from two_layer_net import TwoLayerNet

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch=max_epoch, batch_size=batch_size, eval_interval=10)
trainer.plot()