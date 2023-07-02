import numpy as np
import matplotlib.pyplot as plt

from Perceptron import Perceptron
from Vector import Vector

M_pos = [[0, 1.8], [2, 0.6]]

M_neg = [[-1.2, 1.4], [0.4, -1]]

p = Perceptron(Vector([1, 1]))

w = p.learn(M_pos, M_neg)

history = p.history()

# Vector origin location
X = [0]
Y = [0]

MIN = -3
MAX = 3

# layout the graph
fig, ax = plt.subplots()

# coordinates
ax.spines[["left", "bottom"]].set_position(("data", 0))
ax.spines[["top", "right"]].set_visible(False)
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

ax.set_aspect('equal', adjustable='box')

plt.xlim(MIN, MAX)
plt.ylim(MIN, MAX)



# draw in negative / positive datasets
for set in M_neg:
    c = ax.plot((set[0]), (set[1]), '.', color='r')

for set in M_pos:
    c = ax.plot((set[0]), (set[1]), '.', color='g')


# draw step from first entry in history
step = history[0]
w = step[0]
x = step[1]
scalar = step[2]
result = step[3]


plt.quiver(X, Y, w[0], w[1], color='r', alpha=0.5, scale=1, units='xy', angles='xy', scale_units='xy')
plt.quiver(w[0], w[1],  result[0] - w[0],   result[1] - w[1], ec='black', linestyle='--', linewidth=0.1, alpha=0.5, scale=1, units='xy', angles='xy', scale_units='xy')

# negative/positive plane
plt.plot(
    [MAX * w[0], MIN * w[1]],
    [MIN * w[0], MAX * w[1]]
)

# Show plot with grid
plt.grid()
plt.show()