class Matrix:

    def __init__(self, arr):
        self.arr = arr;

    @staticmethod
    def from_array(arr):

        return Matrix(arr);

        pass

    def mul(self, matrix):

        rt_matrix = matrix.to_array()
        lft_matrix = self.arr

        nl = []

        for _, left_row in enumerate(lft_matrix):
            row = []
            nl.append(row)
            for _, rt_col in enumerate(rt_matrix[0]):
                row.append(0)

        for left_row_idx, left_row in enumerate(lft_matrix):
            for left_col_idx, left_col in enumerate(left_row):
                for rt_col_idx, rt_col in enumerate(rt_matrix[left_col_idx]):
                    nl[left_row_idx][rt_col_idx] += lft_matrix[left_row_idx][left_col_idx] * rt_matrix[left_col_idx][rt_col_idx]
        return Matrix.from_array(nl)


    def to_array(self):
        return self.arr


pass

