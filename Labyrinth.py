import numpy as np
from Layer import *

class NeuralNetwork:
    def __init__(self, dims):
        self.dims = dims

        # Create all the layers:
        self.layers = [Layer(dims[0], 0)]
        for i in range(1, len(dims)):
            self.layers.append(Layer(dims[i], dims[i - 1]))

    # Performs feedforward once b yusing the
    # inputs given and the net's internal state:
    def feedforward(self, input):
        self.layers[0].activ = input
        for i in range(1, len(self.layers)):
            self.layers[i].activate(self.layers[i - 1].activ)

    # Calculates the accuracy of the network by taking in the test dataset
    # and calling the user defined function self.test_f():
    def test(self, test_data):
        correct = 0  # Track the number of correct predictions
        for i in range(len(test_data)):
            self.feedforward(test_data[i])
            if self.test_f(i, self.layers[-1].activ):
                correct += 1
            print(f'Tested: {i + 1} / {len(test_data)}', end='\r')
        print(f'\nCorrect: {correct} / {len(test_data)}')
        print(f'Accuracy: {(correct / len(test_data)) * 100}%')

    # Sets the user defined functions as instance variables
    # to call them later when and where needed:
    def set_routines(self, loss_f, d_loss_f, test_f=None):
        self.loss_f = loss_f
        self.d_loss_f = d_loss_f
        self.test_f = test_f

    # Performs all the training operations!
    # Takes in the training data and labels,
    # the learning rate, and epochs:
    def train(self, train_data, train_labels, eta=1, epochs=100):
        # Loop through all epochs:
        for i in range(epochs):
            # Loop through all the training data:
            for j in range(len(train_data)):
                # Set input, output and target variables:
                input = train_data[j]
                self.feedforward(input)
                output = self.layers[-1].activ
                target = train_labels[j]

                # Compute loss by calling the custom loss function:
                loss = self.loss_f(output, target)  # Not really used yet...
                # Compute the derivative of the Loss function with
                # respect to the activations of the last layer:
                self.layers[-1].activ_gradient = self.d_loss_f(output, target)
                # Loop through and train all the layers in reverse order,
                # using the computed gradient and the learning rate:
                for k in range(len(self.layers) - 1, 0, -1):
                    self.layers[k].tweak_params(self.layers[k - 1], eta)

