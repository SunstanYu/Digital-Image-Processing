import math
from scipy import interpolate
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import time


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


path = "rice.tif"
img = Image.open(path)
print(type(img))
arr = np.array(img)
print(arr.shape)
warnings.filterwarnings("ignore")
test = 2
# a = np.ones([50,50])
dim = [round(arr.shape[0] * 0.7), round(arr.shape[1] * 0.7)]
print(dim)
# print(a[0, 1])
time_start = time.time()
if test == 0:
    b = nearest_yujincheng(arr, dim)
elif test == 1:
    b = bilinear_yujincheng(arr, dim)
elif test == 2:
    b = bicubic_yujincheng(arr, dim)
    b = b.astype(np.uint8)
print(b.shape)

time_end = time.time()
print('time cost', time_end - time_start, 's')
im = Image.fromarray(b)
print(type(im))
# im = im.astype(np.uint8)
# im.save("rice22.png")
plt.imshow(im, cmap="gray")
plt.show()
# x = np.arange(-5.01, 5.01, 0.25)
# y = np.arange(-5.01, 5.01, 0.25)
# xx, yy = np.meshgrid(x, y)
# z = np.sin(xx**2+yy**2)

# f = interpolate.interp2d(x, y, z, kind='cubic')
#
# x = np.arange([1, 2])
# print(x)

# [b, b1] = a.shape
# c = np.zeros([2, 3], dtype=np.uint8)
# c[1, 2]=-1
# print(c)
# print(len(c[0]))
# print(type(b))
# print([b, b1])
# for i in range(0, 2):
#     print(i)
# x = np.ones([2,2], dtype = int)*(-1)
# print(x)
