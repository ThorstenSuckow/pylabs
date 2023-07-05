class Vector:

    def __init__(self, arr):
        self.arr = arr;

    @staticmethod
    def from_array(arr):

        return Vector(arr);

        pass

    def mul(self, vector):

        if isinstance(vector, Vector):
            vector = vector.to_array()

        if isinstance(vector, list):
            if len(self.arr) != len(vector):
                raise Exception("length of this vector not equal to target vector")
            v = 0.0
            for idx, value in enumerate(self.arr):
                v += self.arr[idx] * vector[idx]
            return v

        else:
            f = vector
            vector = []
            for i in self.arr:
                vector.append(f * i)
            return Vector.from_array(vector)


    def add(self, vector):

        if isinstance(vector, Vector):
            vector = vector.to_array()

        if len(self.arr) != len(vector):
            raise Exception("length of this vector not equal to target vector")

        v = []

        for idx, value in enumerate(self.arr):
            v.append(self.arr[idx] + vector[idx])

        return Vector(v)

    def subtract(self, vector):

        if isinstance(vector, Vector):
            vector = vector.to_array()

        if len(self.arr) != len(vector):
            raise Exception("length of this vector not equal to target vector")

        v = []

        for idx, value in enumerate(self.arr):
            v.append(self.arr[idx] - vector[idx])

        return Vector(v)

    def to_array(self):
        return self.arr


pass
