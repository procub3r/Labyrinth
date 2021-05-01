import numpy as np

# Create a random number generator:
rng = np.random.default_rng()

class Layer:
    def __init__(self, size, prev_size, activ_f='sigmoid'):
        self.size = size  # # of nodes in the layer
        self.prev_size = prev_size  # # of nodes in the previous layer
        # Gradient of loss function w.r.t the activations:
        self.activ_gradient = np.zeros(size)
        self.activ = np.zeros(size)  # activations
        self.w_sum = np.zeros(size)  # weighted sum
        # Weights:
        self.weight = rng.uniform(-2, 2,
                size=(size * prev_size)).reshape((size, prev_size))
        self.bias = rng.uniform(-2, 2, size=size)  # biases
        # Set the activation and de-activation functions of the layer:
        self.set_activ_f(activ_f)

    # Sets the activation and de-activation functions of the layer:
    def set_activ_f(self, activ_f_):
        self.activ_f, self.d_activ_f = Layer.activ_funcs[activ_f_]

    # Tweaks the weights and biases to minimize the ultimate loss!
    # Also computes the gradient of the loss function w.r.t the 
    # activations of the previous layer:
    def tweak_params(self, prev_layer, eta):
        d_w_sum = self.activ_gradient * self.d_activ_f(self.w_sum)
        prev_layer.activ_gradient = eta * d_w_sum @ self.weight
        self.weight -= eta * np.outer(d_w_sum, prev_layer.activ)
        self.bias -= eta * d_w_sum

    # Activates the layer
    def activate(self, input):
        self.w_sum = (self.weight @ input) + self.bias
        self.activ = self.activ_f(self.w_sum)

    # Activation and de-activation functions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        return Layer.sigmoid(x) * (1 - Layer.sigmoid(x))


# Lookup table for all the activation functions:
Layer.activ_funcs = {
    'sigmoid': (Layer.sigmoid, Layer.d_sigmoid),
}

