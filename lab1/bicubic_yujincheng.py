from scipy import interpolate
import numpy as np


def bicubic_yujincheng(input_file: np.ndarray, dim: list) -> np.ndarray:
    '''
    :param input_file: list that to be interpolated
    :param dim: dimension of the interpolated image
    :return: interpolated list
    '''
    if input_file.ndim != 2:
        return
    row = np.arange(0, input_file.shape[0], 1)
    col = np.arange(0, input_file.shape[1], 1)
    # input=input_file.flatten(order = 'F')
    f = interpolate.interp2d(col, row, input_file, kind="cubic")
    xx = np.arange(0, input_file.shape[0], input_file.shape[0] / dim[0])
    yy = np.arange(0, input_file.shape[1], input_file.shape[1] / dim[1])
    out = f(xx, yy)
    print(out.shape)
    # out = lambda xx, ynew: r(xnew, ynew).T
    # znew_r = plot(rt, xnew, ynew)
    return out
