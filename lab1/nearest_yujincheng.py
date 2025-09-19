import numpy as np


def nearest_yujincheng(input_file: np.ndarray, dim: list) -> np.ndarray:
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
            # row = round(ratio_row * i)
            # col = round(ratio_col * j)
            out[i, j] = input_file[round(ratio_row * i), round(ratio_col * j)]
    return out
