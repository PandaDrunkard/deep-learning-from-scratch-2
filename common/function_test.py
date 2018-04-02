# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from function import sigmoid

def visualize_sigmoid():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.title('sigmoid')
    plt.show()

visualize_sigmoid()