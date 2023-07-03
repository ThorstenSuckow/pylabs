from Vector import Vector


class Perceptron:

    def __init__(self, vector):

        self._history = None

        if isinstance(vector, list):
            self.w = Vector(vector)

        self.w = vector

    def learn(self, positive_set, negative_set):

        w = self.w

        pos_ok = False
        neg_ok = False

        # w before weight changes, w.x, resulting w
        history = []

        while not neg_ok or not pos_ok:

            for x in positive_set:
                scalar = w.mul(x)
                step = [w.to_array(), Vector(x).to_array(), scalar, w.to_array(), x]
                #print("learning", scalar)
                if scalar <= 0:
                    pos_ok = False
                    w = w.add(x)
                    step[3] = w.to_array()
                    history.append(step)
                    #print("add:", w.to_array())
                else:
                    pos_ok = True

            for x in negative_set:
                scalar = w.mul(x)
                step = [w.to_array(), Vector(x).to_array(), scalar, w.to_array(), x]
                #print("learning", scalar)
                if scalar > 0:
                    neg_ok = False
                    w = w.subtract(x)
                    step[3] = w.to_array()
                    history.append(step)
                    #print("sub:", w.to_array())
                else:
                    neg_ok = True


        self.w = w
        self._history = history;

        return w

    def history(self):
        return self._history;