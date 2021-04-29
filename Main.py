import numpy as np
import Labyrinth as lbr

dims = [2, 2, 1]
nn = lbr.NeuralNetwork(dims)

test_data = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
test_labels = np.array([[0], [0], [1], [1]])

def test_f(i, output):
    if output == test_labels[i]:
        return True

def loss_f(output, target):
    return (output - target) ** 2

def d_loss_f(output, target):
    return 2 * (output - target)

nn.test_f = test_f
nn.loss_f = loss_f
nn.d_loss_f = d_loss_f

# nn.train(test_data, test_labels, epochs=5, eta=0.001)
nn.test(test_data)

