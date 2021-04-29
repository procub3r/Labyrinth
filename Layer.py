import numpy as np

class Layer:
    def __init__(self, size, prev_size, activ_f='sigmoid'):
        self.size = size
        self.prev_size = prev_size
        self.activ_gradient = np.ones(size)
        self.activ = np.ones(size)
        self.w_sum = np.ones(size)
        self.weight = np.ones((size, prev_size))
        self.bias = np.ones(size)
        self.activ_f, self.d_activ_func = Layer.activ_funcs[activ_f]

    def tweak_params(self, prev_layer, eta):
        pass

    def activate(self, input):
        self.w_sum = (self.weight @ input) + self.bias
        self.activ = self.activ_f(self.w_sum)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        return Layer.sigmoid(x) * (1 - Layer.sigmoid(x))

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def d_relu(x):
        return int(x > 0)

# Lookup table for all the activation functions:
Layer.activ_funcs = {
    'sigmoid': (Layer.sigmoid, Layer.d_sigmoid),
    'relu': (Layer.relu, Layer.d_relu)
}

