import cv2
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt


def array_to_hist(arr):
    """
    :param input_image: input image
    :return: corresponding histogram
    """
    # arr = np.array(input_image).astype(np.int32)
    [width, height] = arr.shape
    output_hist = np.zeros(256)
    for line in arr:
        for point in line:
            output_hist[point] += 1
    N = width * height
    output_hist = output_hist / N
    return output_hist


def hist_match_12012223(input_image, hist_desired):
    """
    :param input_image: input image
    :param hist_desired: desired histogram
    :return output_array: output array
    """
    # 计算累计直方图

    hist_after = np.int32(255 * np.cumsum(hist_desired))
    input_hist = array_to_hist(input_image)
    hist_before = np.int32(255 * np.cumsum(input_hist))
    # 计算映射
    M = np.zeros(256, dtype=np.uint8)
    idx = 0
    for i in range(256):
        minv = 1
        for j in hist_after[idx:]:
            if np.fabs(hist_after[j] - hist_before[i]) <= minv:
                minv = np.fabs(hist_after[j] - hist_before[i])
                idx = int(j)
        M[i] = idx
    output_image = Image.fromarray(M[np.array(input_image)])
    out_hist = array_to_hist(output_image)
    return output_image, out_hist, input_hist


# input_img = Image.open("Q3_2.tif")
# dist_img = Image.open("mask1.tif")
# time_start = time.time()
# hist_desired = array_to_hist(dist_img)
# output_image, out_hist, input_hist = hist_match_12012223(input_img, hist_desired)
# time_end = time.time()
# print('time cost', time_end - time_start, 's')
# plt.imshow(output_image, cmap="gray")
# # output_image.save("Q3_2_img_mask1.tif")
# plt.show()
# x = range(len(input_hist))
# plt.bar(x, input_hist * 25, color='black', edgecolor='skyblue')
# x_label = ['{}'.format(i) for i in x]
# plt.xticks(x[::50], x_label[::50])
# plt.xlabel('Intensity')
# plt.ylabel('Number of pixels(X $10^{4}$)')
# plt.title('Histogram of the Input Image')
# # plt.savefig('Q2_mask1_in.png')
# plt.show()
#
# plt.bar(x, hist_desired * 25, color='black', edgecolor='skyblue')
# x_label = ['{}'.format(i) for i in x]
# plt.xticks(x[::50], x_label[::50])
# plt.xlabel('Intensity')
# plt.ylabel('Number of pixels(X $10^{4}$)')
# plt.title('Histogram of the Output Image')
# # plt.savefig('Q2_mask1.png')
# plt.show()
#
# plt.bar(x, out_hist * 25, color='black', edgecolor='skyblue')
# x_label = ['{}'.format(i) for i in x]
# plt.xticks(x[::50], x_label[::50])
# plt.xlabel('Intensity')
# plt.ylabel('Number of pixels(X $10^{4}$)')
# plt.title('Histogram of the Output Image')
# # plt.savefig('Q2_mask1_out.png')
# plt.show()
