import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
from matplotlib.patches import Circle


class PerceptronPlotter:

    def __init__(self, log, X, y, min=-5, max=5, ):

        self.anim = None
        self.next_weight_text = None
        self.previous_weight_text = None
        self.anchored_box = None
        self.current_data_text = None
        self.bias_text = None
        self.epoch_text = None
        self.learning_rate_text = None
        self.w_initial_text = None
        self.accuracy_text = None
        self.active_value = None
        self.separator = None
        self.weight_vector = None
        self.next_weight_vector = None

        self.is_drawn = False

        self.fig, self.ax = plt.subplots()
        self.min = min
        self.max = max
        self.log = log
        self.X = X
        self.y = y

    def draw_graph(self):
        self.is_drawn=True
        ax = self.ax
        fig = self.fig
        MIN = self.min
        MAX = self.max

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

    def draw_data(self):

        ax = self.ax
        X = self.X
        y = self.y

        (n, _) = X.shape
        for i in range(n):
            ax.plot((X[i][0]), (X[i][1]), '.', color='r' if y[i] == 0 else "g")

    def update_plot(self, step):
        ax = self.ax

        w_prev = step["w_prev"]
        X = step["X"]
        w = step["w"]

        if not self.weight_vector:
            self.weight_vector = plt.quiver(
                0, 0, 0, 0,
                color='r', scale=1, units='xy', angles='xy', scale_units='xy', width=0.02
            )

        if not self.next_weight_vector:
            self.next_weight_vector = plt.quiver(
                0, 0, 0, 0,
                ec='black', linestyle='--', scale=1, units='xy', angles='xy', scale_units='xy', width=0.02
            )

        if not self.active_value:
            self.active_value = plt.Circle((X[0], X[1]), 0.1, alpha=0.5, color='orange')
            ax.add_patch(self.active_value)

        if not self.separator:
            self.separator = plt.plot([0, 0], [0, 0], 'k', ls=':')

        self.active_value.center = X

        self.weight_vector.set_UVC(w_prev[0], w_prev[1])
        self.next_weight_vector.set_UVC(w[0], w[1])
        self.next_weight_vector.set_offsets(w_prev)

        # w[0] * x + w[1] * y + bias = 0
        # => y = mx + c: w[1]*y = -w[0]*x - bias
        slope = -w_prev[0] / w_prev[1]
        intercept = -step["bias"] / w_prev[1]
        self.separator[0].set(xdata=[-5, 5], ydata=slope * np.array([-5, 5]) + intercept)

    def update_meta_info(self, step):

        MAX = self.max

        if not self.epoch_text:
            self.epoch_text = plt.text(MAX, -1, "")

        if not self.bias_text:
            self.bias_text = plt.text(MAX, -1.5, "")

        if not self.accuracy_text:
            self.accuracy_text = plt.text(MAX, -2, "")

        if self.learning_rate_text is None:
            self.learning_rate_text = plt.text(MAX, -3, f"Learning Rate: " + str("%.3f" % step['learning_rate']))
        if self.w_initial_text is None:
            self.w_initial_text = plt.text(
                MAX, -3.5,
                f"Initial w: " + str("%.3f" % step['w_initial'][0]) + "  " + str("%.3f" % step['w_initial'][1])
            )

        self.bias_text.set_text(f"Bias: " + str("%.3f" % step['bias']))

        self.accuracy_text.set_text(f"Accuracy: " + str("%.3f" % step['accuracy']))

        epoch = step["epoch"]
        epochs_required = step["epochs_required"]
        self.epoch_text.set_text(f"Epoch: {epoch} / {epochs_required}")

    def update_calculation_info(self, step):
        ax = self.ax

        MAX = self.max

        w_prev = step["w_prev"]
        X = step["X"]
        w = step["w"]

        if not self.next_weight_text:
            self.next_weight_text = plt.text(MAX, 1.5, "", color="k")

        if not self.previous_weight_text:
            self.previous_weight_text = plt.text(MAX, 2.5, "", color="r")

        if not self.current_data_text:
            self.current_data_text = TextArea("", textprops=dict(color="k"))

        self.previous_weight_text.set_text(
            r' $w = \binom{' + str("%.2f" % w_prev[0]) + '}{' + str("%.2f" % w_prev[1]) + '}$')
        self.next_weight_text.set_text(r"$\Delta w=(" + str("%.2f" % w[0]) + ", " + str("%.2f" % w[1]) + ")$")
        self.current_data_text.set_text(r"$x = (" + str("%.2f" % X[0]) + ", " + str("%.2f" % X[1]) + ")$")

        if not self.anchored_box:
            box2 = DrawingArea(0, 0, 0, 0)
            c2 = Circle((0, -5.5), radius=4.5, fc="orange")
            box2.add_artist(c2)
            box = HPacker(children=[box2, self.current_data_text],
                          align="right",
                          pad=0, sep=5)
            self.anchored_box = AnchoredOffsetbox(loc='lower left',
                                             child=box, pad=0.,
                                             frameon=False,
                                             bbox_to_anchor=(1.02, 0.82),
                                             bbox_transform=ax.transAxes,
                                             borderpad=0.,
                                             )
            ax.add_artist(self.anchored_box)

    def frame(self, num, from_anim = False):
        step = self.log[num]

        if not self.is_drawn:
            self.draw_graph()
            self.draw_data()

        self.update_plot(step)
        self.update_calculation_info(step)
        self.update_meta_info(step)

        if not from_anim:
            plt.show();

    def init_anim(self):
        pass


    def animate(self):

        self.frame(0, True)

        anim = animation.FuncAnimation(
            self.fig, self.frame,
            init_func=self.init_anim,
            fargs=([True]),
            frames=len(self.log),
            interval=1000,
            blit=False)

        plt.show()

        return anim
