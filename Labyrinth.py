import numpy as np
from Layer import *

class NeuralNetwork:
    def __init__(self, dims):
        self.dims = dims
        self.layers = []
        for i in range(1, len(dims)):
            self.layers.append(Layer(dims[i], dims[i - 1]))

    def feedforward(self, input):
        self.layers[0].activate(input)
        for i in range(1, len(self.layers)):
            self.layers[i].activate(self.layers[i - 1].activ)

    def test(self, test_data, test_f):
        correct = 0
        for i in range(len(test_data)):
            if test_f(i, self.layers[-1].activ):
                correct += 1
            print(f'Tested: {i + 1} / {len(test_data)}', end='\r')
        print(f'\nAccuracy: {(correct / len(test_data)) * 100}% --- + Correct: {correct} / {len(test_data)}')
