import numpy as np

class Layer:
    def __init__(self, size, prev_size):
        self.size = size
        self.prev_size = prev_size
        self.activ = np.ones(size)
        self.w_sum = np.ones(size)
        self.weight = np.ones((size, prev_size))
        self.bias = np.ones(size)
        self.activ_f = Layer.relu

    def activate(self, input):
        self.w_sum = (self.weight @ input) + self.bias
        self.activ = self.activ_f(self.w_sum)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return x * (x > 0)

