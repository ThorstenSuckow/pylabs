from Matrix import Matrix

def test_mul():

    m1 = Matrix.from_array([
        [1, 1]
    ]);

    m2 = Matrix.from_array([
        [-1.2],
        [1.4]
    ]);

    assert m1.mul(m2).to_array() == [[0.19999999999999996]]

    m1 = Matrix.from_array([
        [3, 2, 1],
        [1, 0, 2]
    ]);

    m2 = Matrix.from_array([
        [1, 2],
        [0, 1],
        [4, 0]
    ]);

    assert m1.mul(m2).to_array()== [
        [7, 8],
        [9, 2]
    ]

    m1 = Matrix([
        [0, 2],
        [0, 0]
    ]);

    m2 = Matrix([
        [0, 0],
        [2, 0]
    ]);

    assert m1.mul(m2).to_array() == [
        [4, 0],
        [0, 0]
    ]

    m1 = Matrix([
        [0, 2, 0, 0]
    ]);

    m2 = Matrix([
        [2],
        [3],
        [4],
        [5]
    ]);

    assert m1.mul(m2).to_array() == [
        [6]
    ]

    m1 = Matrix([
        [2],
        [3],
        [4],
        [5]
    ]);

    m2 = Matrix([
        [0, 2, 0, 0]
    ]);

    assert m1.mul(m2).to_array() == [
        [0, 4, 0, 0],
        [0, 6, 0, 0],
        [0, 8, 0, 0],
        [0, 10, 0, 0]
    ]

    pass