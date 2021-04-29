import numpy as np
from Layer import *

class NeuralNetwork:
    def __init__(self, dims):
        self.dims = dims
        self.layers = []
        for i in range(1, len(dims)):
            self.layers.append(Layer(dims[i], dims[i - 1], 'sigmoid'))
        self.test_f = None
        self.loss_f = None
        self.d_loss_f = None

    def feedforward(self, input):
        self.layers[0].activate(input)
        for i in range(1, len(self.layers)):
            self.layers[i].activate(self.layers[i - 1].activ)
        print(self.layers[-1].activ)

    def test(self, test_data):
        correct = 0
        for i in range(len(test_data)):
            if self.test_f(i, self.layers[-1].activ):
                correct += 1
            print(f'Tested: {i + 1} / {len(test_data)}', end='\r')
        print(f'\nCorrect: {correct} / {len(test_data)}')
        print(f'Accuracy: {(correct / len(test_data)) * 100}%')

    def train(self, train_data, train_labels, eta=1, epochs=100):
        for i in range(epochs):
            for j in range(len(train_data)):
                input = train_data[j]
                self.feedforward(input)
                output = self.layers[-1].activ
                target = train_labels[j]
                loss = self.loss_f(output, target)
                self.layers[-1].activ_gradient = self.d_loss_f(output, target)
                for k in range(len(self.layers) - 1, 0, -1):
                    self.layers[k].tweak_params(self.layers[k - 1], eta)
