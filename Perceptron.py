import numpy


class Perceptron:

    def __init__(self, n_epochs=10, learning_rate=1, w=None):

        self.log = []
        self.result_info = None
        self.epoch_list = None
        self.w = None
        self.learning_rate = None
        self.bias = None
        self.epochs = None

        self.config = {
            "epochs": n_epochs,
            "learning_rate": learning_rate,
            "w": w,
            "bias": 0
        }

    def test(self, X):

        z = X @ self.w + self.bias

        # 0 z < 0
        # 1 z == 0
        # 1 z > 0
        return numpy.heaviside(z, 1)

    def learn(self, X, y):

        self.apply_defaults()

        # s sampled
        # m features
        (n, m) = X.shape  # n is the number of samples, m is the number of features

        self.w = numpy.random.randn(m) if self.w is None else self.w

        w_initial = self.w

        epochs = self.epochs

        learning_rate = self.learning_rate

        errors = 0

        for epoch in range(epochs):

            errors = 0

            for i in range(n):
                expected = y[i]
                result = self.test(X[i])
                error = 0

                #   1 != 0
                #   0 != 1
                w_prev = self.w.copy()
                if result != expected:
                    error = expected - result

                    self.w += (X[i] * learning_rate * error)
                    self.bias += learning_rate * error

                    errors += 1

                self.log.append({
                    "w": self.w.copy(),
                    "w_prev": w_prev,
                    "epoch": f"{epoch + 1}.{i + 1}",
                    "accuracy": 1 - (errors / n),
                    "bias": self.bias,
                    "error": error,
                    "X": X[i],
                    "y": y[i],
                    "result": result,
                    "epochs_required": -1,
                    "learning_rate": learning_rate,
                    "w_initial": w_initial
                })
            accuracy = 1 - (errors / n)
            self.epoch_list.append(f'Epoch {epoch + 1}: accuracy = {accuracy:.3f}')

            # if epochs are high enough, we expect to exit the training early
            # there should be a few epochs remaining, see result_info
            if errors == 0:
                break

        self.result_info = f'Epochs ({epoch + 1}/{self.epochs}) finished, w is = {self.w}, bias is {self.bias} error count is {errors:.3f}'

        for log in self.log:
            log["epochs_required"] = epoch + 1

        return self.w

    def apply_defaults(self):
        self.epoch_list = []
        self.bias = self.config["bias"]
        self.learning_rate = self.config["learning_rate"]
        self.epochs = self.config["epochs"]
        self.w = self.config["w"]
        self.log = []
