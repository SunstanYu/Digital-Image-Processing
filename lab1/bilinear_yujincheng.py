import math
import numpy as np


def bilinear_yujincheng(input_file: np.ndarray, dim: list) -> np.ndarray:
    '''
    :param input_file: list that to be interpolated
    :param dim: dimension of the interpolated image
    :return: interpolated list
    '''
    if input_file.ndim != 2:
        return
    [row, col] = input_file.shape
    ratio_row = row / dim[0]
    ratio_col = col / dim[1]
    out = np.zeros([dim[0], dim[1]], dtype=np.uint8)
    for i in range(0, dim[0]):
        for j in range(0, dim[1]):
            # row_num = ratio_row * i
            row_num = ratio_row * (i + 0.5) - 0.5
            # col_num = ratio_col * j
            col_num = ratio_col * (j + 0.5) - 0.5
            if row_num < 0:
                row_num = 0
            elif row_num > row - 1:
                row_num = row - 1
            if col_num < 0:
                col_num = 0
            elif col_num > col - 1:
                col_num = col - 1

            r1 = math.floor(row_num)
            r2 = math.ceil(row_num)
            c1 = math.floor(col_num)
            c2 = math.ceil(col_num)

            bilinear1 = input_file[r1, c1] * (c2 - col_num) + input_file[r1, c2] * (1 - (c2 - col_num))
            bilinear2 = input_file[r2, c1] * (c2 - col_num) + input_file[r2, c2] * (1 - (c2 - col_num))
            bilinear3 = bilinear1 * (r2 - row_num) + bilinear2 * (1 - (r2 - row_num))
            out[i, j] = bilinear3
    return out
