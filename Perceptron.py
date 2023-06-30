from Vector import Vector


class Perceptron:

    def __init__(self, vector):

        if isinstance(vector, list):
            self.w = Vector(vector)

        self.w = vector
        pass

    def learn(self, positive_set, negative_set):

        w = self.w

        pos_ok = False
        neg_ok = False

        while not neg_ok or not pos_ok:

            for x in positive_set:
                scalar = w.mul(x)
                print("learning", scalar)
                if scalar <= 0:
                    pos_ok = False
                    w = w.add(x)
                    print("add:", w.to_array())
                else:
                    pos_ok = True

            for x in negative_set:
                scalar = w.mul(x)
                print("learning", scalar)
                if scalar > 0:
                    neg_ok = False
                    w = w.subtract(x)
                    print("sub:", w.to_array())
                else:
                    neg_ok = True

        self.w = w

        return w
