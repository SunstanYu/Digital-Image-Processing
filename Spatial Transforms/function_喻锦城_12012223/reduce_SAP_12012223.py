import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def reduce_SAP_12012223(input_image, n_size):
    arr = np.array(input_image)
    output_median = arr.copy()
    output_mean = arr.copy()
    # 将图像分割成指定大小的块
    height, width = arr.shape
    seg_height, seg_width = n_size, n_size
    rows = height // seg_height
    cols = width // seg_width
    # 对每个块进行局部直方图均衡化
    for i in range(rows):
        for j in range(cols):
            y = i * seg_height
            x = j * seg_width
            seg = arr[y:y + seg_height, x:x + seg_width]
            output_median[y:y + seg_height, x:x + seg_width] = np.median(seg)
            # print(np.median(seg))
            output_mean[y:y + seg_height, x:x + seg_width] = np.mean(seg)
            # print(np.mean(seg))
    output_median_image = Image.fromarray(output_median)
    output_mean_image = Image.fromarray(output_mean)
    return output_median_image, output_mean_image

#
# input_img = Image.open("Q3_4.tif")
# time_start = time.time()
# output_median, output_mean = reduce_SAP_12012223(input_img, 5)
# time_end = time.time()
# print('time cost', time_end - time_start, 's')
# plt.imshow(output_median, cmap="gray")
# # output_median.save("Q4_median_5.png")
# plt.show()
# plt.imshow(output_mean, cmap="gray")
# # output_mean.save("Q4_mean_3.png")
# plt.show()
# # 创建一个 1x2 的图窗，并在每个子图中显示一张图片
# # fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# # axs[0].imshow(output_median, cmap="gray")
# # axs[1].imshow(output_mean, cmap="gray")
# #
# # # 设置子图的标题
# # axs[0].set_title('output_median')
# # axs[1].set_title('output_mean')
#
# # 显示图窗
# plt.show()
