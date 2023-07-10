import matplotlib
import matplotlib.pyplot
import numpy as np
from sklearn.datasets import make_blobs

from Perceptron import Perceptron
from PerceptronPlotter import PerceptronPlotter

matplotlib.use("TkAgg")

title = "";

X = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1]
])
# AND
#title= "\"AND\""
#y = np.array([0, 0, 0, 1])
# OR
#y = np.array([0, 1, 1, 1])
# XOR
title = "XOR"
y = np.array([0, 1, 1, 0])

# test with clusters
#X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=2)


p = Perceptron(50, 0.3)
p.learn(X, y)

plotter = PerceptronPlotter(p.log, X, y, title)

anim = plotter.animate(500)
