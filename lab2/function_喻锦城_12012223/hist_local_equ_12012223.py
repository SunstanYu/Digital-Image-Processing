import time

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def local_histogram_processing(img, seg_size):
    """
    使用局部直方图处理算法处理输入的图像

    :param img: 输入的图像
    :param seg_size: 分割图像的尺度
    :return: 处理后的图像
    """
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算图像的均值和标准差
    mean, std = cv2.meanStdDev(gray)

    # 将图像分割成指定大小的块
    height, width = gray.shape
    seg_height, seg_width = seg_size, seg_size
    rows = height // seg_height
    cols = width // seg_width

    # 对每个块进行局部直方图均衡化
    for i in range(rows):
        for j in range(cols):
            y = i * seg_height
            x = j * seg_width
            seg = gray[y:y + seg_height, x:x + seg_width]
            seg_mean = np.mean(seg)
            seg_std = np.std(seg)
            if seg_std > 0:
                seg = (seg - seg_mean) * (std / seg_std) + mean
            gray[y:y + seg_height, x:x + seg_width] = seg

    # 将处理后的灰度图转换回彩色图像
    result = gray

    return result


def array_to_hist(arr):
    """
    :param input_image: input image
    :return: corresponding histogram
    """

    [width, height] = arr.shape
    output_hist = np.zeros(256)
    for line in arr:
        for point in line:
            output_hist[point] += 1
    N = width * height
    output_hist = output_hist / N
    return output_hist


def hist_local_equ_12012223(input_image, m_size):
    arr = np.array(input_image)
    output_arr = arr.copy()
    # 将图像分割成指定大小的块
    height, width = arr.shape
    seg_height, seg_width = m_size, m_size
    rows = height // seg_height
    cols = width // seg_width
    # 对每个块进行局部直方图均衡化
    for i in range(rows):
        for j in range(cols):
            y = i * seg_height
            x = j * seg_width
            seg = arr[y:y + seg_height, x:x + seg_width]
            output_arr[y:y + seg_height, x:x + seg_width] = hist_equ(seg)
    input_hist = array_to_hist(arr)
    output_hist = array_to_hist(output_arr)
    output_image = Image.fromarray(output_arr)
    return output_image, output_hist, input_hist


def hist_equ(arr):
    '''

    :param input_image:
    :return output_image, output_hist, input_hist:
    '''
    output_arr = arr.copy()
    [width, height] = arr.shape
    L = arr.max().item() + 1
    # output_hist = input_hist = np.zeros(256)  # float64
    input_hist = array_to_hist(arr)
    Trans = np.int32(255 * np.cumsum(input_hist))
    for i in range(0, width):
        for j in range(0, height):
            output_arr[i, j] = Trans[arr[i, j]]
    return output_arr


# img = cv2.imread('Q3_3.tif')
#
# # 执行局部直方图处理算法
# result = local_histogram_processing(img, 3)
#
# # 显示处理后的图像
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# input_img = Image.open("Q3_3.tif")
# time_start = time.time()
# output_image, out_hist, input_hist = hist_local_equ_12012223(input_img, 5)
# time_end = time.time()
# print('time cost', time_end - time_start, 's')
# plt.imshow(output_image, cmap="gray")
# # output_image.save("Q3_3_local.png")
# plt.show()
# x = range(len(input_hist))
# plt.bar(x, input_hist * 25, color='black', edgecolor='skyblue')
# x_label = ['{}'.format(i) for i in x]
# plt.xticks(x[::50], x_label[::50])
# plt.xlabel('Intensity')
# plt.ylabel('Number of pixels(X $10^{4}$)')
# plt.title('Histogram of the Input Image')
# # plt.savefig('Q3_input.png')
# plt.show()
#
# plt.bar(x, out_hist * 25, color='black', edgecolor='skyblue')
# x_label = ['{}'.format(i) for i in x]
# plt.xticks(x[::50], x_label[::50])
# plt.xlabel('Intensity')
# plt.ylabel('Number of pixels(X $10^{4}$)')
# plt.title('Histogram of the Output Image')
# # plt.savefig('Q3_output.png')
# plt.show()
