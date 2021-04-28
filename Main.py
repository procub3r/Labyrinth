import numpy as np
import Labyrinth as lbr

dims = [2, 2, 1]
nn = lbr.NeuralNetwork(dims)

x = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
t = np.array([[0], [0], [1], [1]])

def test_f(i, output):
    if output == t[i]:
        return True
    return False

nn.test(x, test_f)

