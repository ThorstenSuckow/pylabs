import random

from Vector import Vector
import numpy


class Perceptron2:

    def __init__(self):

        self._history = None
        self.b = 0
        self.bias = 0


    def test(self, X):

        # @ == matrix multiplication
        X_n = numpy.append(X, 1)
        w_n = numpy.append(self.w, self.bias)

        z = X_n @ w_n
        #z = X @ self.w + self.bias

        # 0 z < 0
        # 1 z == 0
        # 1 z > 0
        return numpy.heaviside(z, 1)


    def learn(self, X, y):

        (n, m) = X.shape  # n is the number of samples, m is the number of features

        self.w = numpy.random.randn(m)

        epochs = 50
        self.bias = 0
        alpha = 0.6

        for epoch in range(epochs):

            errors = 0

            for i in range(n):
                expected = y[i]
                result = self.test(X[i])

                #   1 != 0
                #   0 != 1
                if result != expected:
                    error = expected - result

                    self.w += (X[i] * alpha * error)
                    self.bias += alpha * error

                    errors += 1
                    print(X[i], self.w, self.bias, result, expected, (X[i] * alpha * error))

            accuracy = 1 - (errors / n)
            print(f'Epoch {epoch + 1}: accuracy = {accuracy:.3f}')

            if errors == 0:
                break

        print(f'Epochs ({epoch + 1}) finished, w is = {self.w}, bias is {self.bias} error count is {errors:.3f}')


        return self.w

