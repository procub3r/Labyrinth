import numpy as np
import Labyrinth as lbr  # Import the library

dims = [2, 2, 1]  # Dimensions of the network
nn = lbr.NeuralNetwork(dims)  # Create the network

'''
This is a demo program which demonstrates the
neural network by training it on the XOR problem, 
where the output should be 1 if one and only one
of the two inputs equals 1, else 0.
'''

# Test dataset, contains 4 input arrays, each of length 2:
test_data = np.array([[1, 0],  # gives 1
                      [0, 0],  # gives 0
                      [1, 1],  # gives 1
                      [0, 1]])  # gives 0

# Labels for the test dataset which specify the target
# of the network given inputs from the test dataset:
test_labels = np.array([[1],  # target of inputs specified in line #14
                        [0],  # target of inputs specified in line #15
                        [0],  # target of inputs specified in line #16
                        [1]])  # target of inputs specified in line #17

# Test function, which will determine if the output of
# the network is correct or not.
def test_f(i, output):
    # Check if both the output and the target are above 0.5, or below 0.5:
    if (output[0] > 0.5) == (test_labels[i, 0] > 0.5):
        return True  # return True to consider a target match.

# Loss function used by the network to compute the loss of the network:
def loss_f(output, target):
    # Root mean squared error:
    # [ Not really apt for this use-case, but does
    # a good job of demonstrating the lib ;) ]
    return (output - target) ** 2

# Derivative of the loss function used.
def d_loss_f(output, target):
    return 2 * (output - target)

# Set the functions used by the network to calculate the loss, derivative of
# loss and to test itself by using set_routines(). These functions *must* be
# defined by you before using the network:
nn.set_routines(loss_f, d_loss_f, test_f)

# Train the network! epochs = # of times the net trains on the training dataset
# and eta = learning rate.
nn.train(test_data, test_labels, epochs=400, eta=1)

# And finally, test the network to find out how well it performs on the
# testing dataset!
nn.test(test_data)

# Print the activations of the last layer (outputs)
# for each element in the training dataset:
for i in test_data:
    nn.feedforward(i)
    print(nn.layers[-1].activ)

# NOTE: The training dataset and the testing dataset are the same in this case.
# This should not be the case in a real world scenario!
