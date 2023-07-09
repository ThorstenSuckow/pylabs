import matplotlib
import matplotlib.pyplot
import numpy as np
from Perceptron import Perceptron
from PerceptronPlotter import PerceptronPlotter

matplotlib.use("TkAgg")


X = np.array([
    [1, 1], [2, 2], [3, 4], [4, 4]
])
y = np.array([0, 0, 1, 1])

MIN = -5
MAX = 5

p = Perceptron(50, 0.3)
p.learn(X, y)

plotter = PerceptronPlotter(p.log, X, y)

anim = plotter.animate()
