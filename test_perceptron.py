from Perceptron import Perceptron
from Vector import Vector

def test_learn():

    M_pos = [
        [0, 1.8], [2, 0.6]
    ]

    M_neg = [
        [-1.2, 1.4], [0.4, -1]
    ]

    p = Perceptron(Vector([1, 1]))

    w = p.learn(M_pos, M_neg)

    history = p.history()

    for step in history:
        print(step)
