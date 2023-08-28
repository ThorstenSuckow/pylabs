import math

import matplotlib
import matplotlib.pyplot
import numpy as np
from sklearn.datasets import make_blobs

from Perceptron import Perceptron
from PerceptronPlotter import PerceptronPlotter

matplotlib.use("WebAgg")

title = "";

X = np.array([
    [1, 1], [1, 1.2]
])
y =  np.array(
    [0, 1]
)
# AND
title= "\"AND\""
X = np.array([
    [0, 0], [1, 0], [0, 1], [1,1]
])
y = np.array([0, 0, 0, 1])
# OR
#title= "\"OR\""
#y = np.array([0, 1, 1, 1])
# XOR
#title = "XOR"
#y = np.array([0, 1, 1, 0])

# test with clusters
title = "Clusters"
X, y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=2.5)


p = Perceptron(50, 1, [1, 1])
p.learn(X, y)

for i in p.tbl:
    print(' '.join(map(str, i)))

plotter = PerceptronPlotter(p.log, X, y, title, None, [3,3])

# comment this line in case you want to save plots to individual files
anim = plotter.animate(100)

#  uncomment to save plots to individual files
#for i in range(p.log):
    #anim = plotter.frame(i, True)
    #fname = "epoch_" + str(math.floor(i/4) + 1) + "_" + str((i % 4) + 1);
    #anim.savefig(".plots/" + fname)