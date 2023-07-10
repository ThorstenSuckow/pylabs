import numpy

from Perceptron import Perceptron
from Vector import Vector


def test_learn():
    epochs = 20
    X = numpy.array([
        [1, 1], [2, 2], [3, 4], [4, 4]
    ])

    y = numpy.array([0, 0, 1, 1])

    p = Perceptron(epochs, 1, [1, 1])

    assert p.config["w"] == [1, 1]
    assert p.config["learning_rate"] == 1
    assert p.config["epochs"] == epochs
    assert p.config["bias"] == 0

    w = p.learn(X, y)

    assert w.tolist() == [-2.0, 4.0]
    assert p.w is w
    assert p.bias == -8.0
    assert p.learning_rate == 1
    assert p.epochs == epochs

    assert p.test([1, 1.5]) == 0
    assert p.test([2, 3.5]) == 1

    # XOR
    X = numpy.array([
        [0, 0], [0, 1], [1, 0], [1, 1]
    ])
    y = numpy.array([0, 1, 1, 0])
    w = p.learn(X, y)
    assert w is None