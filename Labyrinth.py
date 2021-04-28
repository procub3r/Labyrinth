import numpy as np

class Layer:
    def __init__(self, size, prev_size):
        self.size = size
        self.prev_size = prev_size

class NeuralNetwork:
    def __init__(self, dims):
        self.dims = dims
        self.layers = [Layer(dims[i], dims[i - 1]) for i in range(1, len(dims))]
        for i in self.layers:
            print(i.size)
