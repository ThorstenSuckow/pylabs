import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
from matplotlib.patches import Circle, Ellipse

from Perceptron import Perceptron
from Vector import Vector

M_pos = [[0, 1.8], [2, 0.6]]

M_neg = [[-1.2, 1.4], [0.4, -1]]

p = Perceptron(Vector([1, 1]))

w = p.learn(M_pos, M_neg)

history = p.history()

for entry in history:
    print(entry)

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


def draw_plot(step):
    # draw in negative / positive datasets
    for set in M_neg:
        c = ax.plot((set[0]), (set[1]), '.', color='r')

    for set in M_pos:
        c = ax.plot((set[0]), (set[1]), '.', color='g')

    # draw step from first entry in history
    w = step[0]

    plt.quiver(X, Y, w[0], w[1], color='r', scale=1, units='xy', angles='xy', scale_units='xy', width=0.02)
    plt.text(3, 2.5, r' $w = \binom{' + str("%.2f" % w[0]) + '}{' + str("%.2f" % w[1]) + '}$', color="r")

    if len(step) > 1:
        x = step[1]
        scalar = step[2]
        result = step[3]
        processedset = step[4]

        plt.quiver(w[0], w[1], result[0] - w[0], result[1] - w[1], ec='black', linestyle='--', scale=1, units='xy',
                   angles='xy', scale_units='xy', width=0.02)

        if processedset:
            c1 = plt.Circle((processedset[0], processedset[1]), 0.1, alpha=0.5, color='orange')
            ax.add_patch(c1)
            # data from first set

            box1 = TextArea( r"$x = (" + str("%.2f" % processedset[0]) + ", " + str("%.2f" % processedset[1]) + ")$", textprops=dict(color="k"))
            box2 = DrawingArea(0, 0, 0, 0)
            c2 = Circle((0, -5.5), radius=4.5, fc="orange")
            box2.add_artist(c2)
            box = HPacker(children=[box2, box1],
                          align="right",
                          pad=0, sep=5)
            anchored_box = AnchoredOffsetbox(loc='lower left',
                                             child=box, pad=0.,
                                             frameon=False,
                                             bbox_to_anchor=(1.02, 0.82),
                                             bbox_transform=ax.transAxes,
                                             borderpad=0.,
                                             )
            ax.add_artist(anchored_box)


        # result
        plt.text(3, 1.5, r"$\Delta w=(" + str("%.2f" % result[0]) + ", " + str("%.2f" % result[1]) + ")$")

    # negative/positive plane
    plt.plot(
        # x_1, x_2, y_1, y_2
        [MAX * (w[1] / (0 - w[0])), MIN * (w[1] / (0 - w[0]))], [MAX * 1, MIN * 1]
    )
    fig.subplots_adjust(right=0.7)

    # Show plot with grid
    plt.grid()
    plt.show()


#draw_plot(history[0])
#draw_plot(history[1])
#draw_plot(history[2])

#draw_plot(history[3])
#draw_plot(history[4])

draw_plot([w.to_array()])
