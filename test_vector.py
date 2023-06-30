import unittest
from Vector import Vector


class TestVector(unittest.TestCase):

    def test_mul(self):

        m1 = Vector.from_array([1, 1]);
        assert m1.mul([1, 1]) == 2.0

        m1 = Vector([2, 3, 4, 5]);
        assert m1.mul([2, 2, 2, 3]) == 33


        m1 = Vector([1]);
        assert m1.mul([1]) == 1


    def test_add(self):

        m1 = Vector.from_array([1, 1]);
        assert m1.add([1, 1]).to_array() == [2, 2]

        m1 = Vector([2, 3, 4, 5])
        assert m1.add([2, 2, 2, 3]).to_array() == [4, 5, 6, 8]


    def test_subtract(self):

        m1 = Vector.from_array([1, 1]);
        assert m1.subtract([1, 4]).to_array() == [0, -3]

        m1 = Vector([2, 3, 4, 5])
        assert m1.subtract([2, 2, 2, 3]).to_array() == [0, 1, 2, 2]


    def test_mul_exception(self):
        m1 = Vector([1]);

        self.assertRaises(Exception, m1.mul, [1, 2])
