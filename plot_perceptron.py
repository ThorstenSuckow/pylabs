import matplotlib
import matplotlib.pyplot
import numpy as np
from matplotlib import animation
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
from matplotlib.patches import Circle
from Perceptron import Perceptron
import matplotlib.pyplot as plt
from PerceptronPlotter import PerceptronPlotter

matplotlib.use("TkAgg")
"""
X = np.array([
    [1, 1], [2, 2], [3, 4], [4, 4]
])
y = np.array([0, 0, 1, 1])

MIN = -5
MAX = 5

p = Perceptron(50, 0.3)
p.learn(X, y)

fig, ax = plt.subplots()
file_index = 0

"""

"""
weight_vector = None
vector = None
step_ax = None
next_weight_vector = None
current_data_text = None
next_weight_text = None
previous_weight_text = None
separator = None
active_value = None
epoch_text = None
bias_text = None
accuracy_text = None
learning_rate_text = None
w_initial_text = None
anchored_box = None


def draw_graph():
    # layout the graph
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_aspect('equal', adjustable='box')

    plt.xlim(MIN, MAX)
    plt.ylim(MIN, MAX)
    plt.grid()

    fig.subplots_adjust(right=0.7)


def draw_data(X):
    global active_value
    (n, _) = X.shape
    for i in range(n):
        ax.plot((X[i][0]), (X[i][1]), '.', color='r' if y[i] == 0 else "g")


def update_plot(step):
    global weight_vector
    global next_weight_vector
    global box1
    global next_w_text
    global prev_w_text
    global separator
    global active_value

    w_prev = step["w_prev"]
    bias = step["bias"]
    X = step["X"]
    w = step["w"]

    if not weight_vector:
        weight_vector = plt.quiver(
            0, 0, 0, 0,
            color='r', scale=1, units='xy', angles='xy', scale_units='xy', width=0.02
        )

    if not next_weight_vector:
        next_weight_vector = plt.quiver(
            0, 0, 0, 0,
            ec='black', linestyle='--', scale=1, units='xy', angles='xy', scale_units='xy', width=0.02
        )

    if not active_value:
        active_value = plt.Circle((X[0], X[1]), 0.1, alpha=0.5, color='orange')
        ax.add_patch(active_value)

    if not separator:
        separator = plt.plot([0, 0], [0, 0], 'k', ls=':')

    active_value.center = X

    weight_vector.set_UVC(w_prev[0], w_prev[1])
    next_weight_vector.set_UVC(w[0], w[1])
    next_weight_vector.set_offsets(w_prev)

    # w[0] * x + w[1] * y + bias = 0
    # => y = mx + c: w[1]*y = -w[0]*x - bias
    slope = -w_prev[0] / w_prev[1]
    intercept = -step["bias"] / w_prev[1]
    separator[0].set(xdata=[-5, 5], ydata=slope * np.array([-5, 5]) + intercept)


def update_meta_info(step):
    global accuracy_text
    global bias_text
    global epoch_text
    global learning_rate_text
    global w_initial_text

    if not epoch_text:
        epoch_text = plt.text(MAX, -1, "")

    if not bias_text:
        bias_text = plt.text(MAX, -1.5, "")

    if not accuracy_text:
        accuracy_text = plt.text(MAX, -2, "")

    if learning_rate_text is None:
        learning_rate_text = plt.text(MAX, -3, f"Learning Rate: " + str("%.3f" % step['learning_rate']))
    if w_initial_text is None:
        w_initial_text = plt.text(
            MAX, -3.5, f"Initial w: " + str("%.3f" % step['w_initial'][0]) + "  " + str("%.3f" % step['w_initial'][1])
        )

    bias_text.set_text(f"Bias: " + str("%.3f" % step['bias']))

    accuracy_text.set_text(f"Accuracy: " + str("%.3f" % step['accuracy']))

    epoch = step["epoch"]
    epochs_required = step["epochs_required"]
    epoch_text.set_text(f"Epoch: {epoch} / {epochs_required}")


def update_calculation_info(step):
    global weight_vector
    global next_w
    global current_data_text
    global next_weight_text
    global previous_weight_text
    global separator
    global active_value
    global epoch_text
    global bias_text
    global anchored_box

    w_prev = step["w_prev"]
    bias = step["bias"]
    X = step["X"]
    w = step["w"]

    if not next_weight_text:
        next_weight_text = plt.text(MAX, 1.5, "", color="r")

    if not previous_weight_text:
        previous_weight_text = plt.text(MAX, 2.5, "", color="r")

    if not current_data_text:
        current_data_text = TextArea("", textprops=dict(color="k"))

    previous_weight_text.set_text(r' $w = \binom{' + str("%.2f" % w_prev[0]) + '}{' + str("%.2f" % w_prev[1]) + '}$')
    next_weight_text.set_text(r"$\Delta w=(" + str("%.2f" % w[0]) + ", " + str("%.2f" % w[1]) + ")$")
    current_data_text.set_text(r"$x = (" + str("%.2f" % X[0]) + ", " + str("%.2f" % X[1]) + ")$")

    if not anchored_box:
        box2 = DrawingArea(0, 0, 0, 0)
        c2 = Circle((0, -5.5), radius=4.5, fc="orange")
        box2.add_artist(c2)
        box = HPacker(children=[box2, current_data_text],
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


log = p.log

draw_graph()
draw_data(X)
update_calculation_info(log[0])
update_plot(log[0])
update_meta_info(log[0])


def animate(num):
    step = log[num]

    update_plot(step)
    update_calculation_info(step)
    update_meta_info(step)

    return plt


def init_f():
    global weight_vector
    return weight_vector

"""


def init_f():
    pass


X = np.array([
    [1, 1], [2, 2], [3, 4], [4, 4]
])
y = np.array([0, 0, 1, 1])

MIN = -5
MAX = 5

p = Perceptron(50, 0.3)
p.learn(X, y)

plotter = PerceptronPlotter(p.log, X, y)

plotter.frame(6)

#anim = plotter.animate()


# for idx,step in enumerate(log):
# plt.pause(0.1)


# plt.pause(0.1)

#  continue;

# ax.clear()


# step_idx = 8
# draw_data(X, log[step_idx]["X"])
# draw_plot(log[step_idx])
# draw_plot(history[1])
# draw_plot(history[2])

# draw_plot(history[3])
# draw_plot(history[4])

# draw_plot([w.tolist()])
