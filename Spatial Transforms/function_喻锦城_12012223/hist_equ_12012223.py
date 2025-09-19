import math
from scipy import interpolate
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import time


def hist_equ_12012223(input_image):
    '''

    :param input_image:
    :return output_image, output_hist, input_hist:
    '''
    arr = np.array(input_image)
    output_arr = arr.copy()
    [width, height] = arr.shape
    L = arr.max().item() + 1
    print(type(L))
    output_hist = input_hist = np.zeros(256)  # float64
    print(input_hist.shape)
    for line in arr:
        for point in line:
            input_hist[point] += 1
    N = width * height
    pr = input_hist / N
    Trans = np.int32(255 * np.cumsum(pr))
    for i in range(0, width):
        for j in range(0, height):
            output_arr[i, j] = Trans[arr[i, j]]
    output_image = Image.fromarray(output_arr)
    for line in output_arr:
        for point in line:
            output_hist[point] += 1
    N = width * height
    output_hist = output_hist / N
    return output_image, output_hist, pr


path = "heatmap2.jpg"
img = Image.open(path)
time_start = time.time()
out_img, out_hist, input_hist = hist_equ_12012223(img)
time_end = time.time()
print('time cost', time_end - time_start, 's')
# img_array = np.array(out_img)
plt.imshow(out_img, cmap="gray")
# out_img.save("Q3_1_1_img.png")
plt.show()
# x = range(len(input_hist))
# plt.bar(x, input_hist * 25, color='black', edgecolor='skyblue')
# x_label = ['{}'.format(i) for i in x]
# plt.xticks(x[::50], x_label[::50])
# plt.xlabel('Intensity')
# plt.ylabel('Number of pixels(X $10^{4}$)')
# plt.title('Histogram of the Input Image')
# # plt.savefig('Q1_1_in.png')
# plt.show()
#
# plt.bar(x, out_hist * 25, color='black', edgecolor='skyblue')
# x_label = ['{}'.format(i) for i in x]
# plt.xticks(x[::50], x_label[::50])
# plt.xlabel('Intensity')
# plt.ylabel('Number of pixels(X $10^{4}$)')
# plt.title('Histogram of the Output Image')
# # plt.savefig('Q1_1_out.png')
# plt.show()
#
# path = "Q3_1_2.tif"
# img = Image.open(path)
# out_img, out_hist, input_hist = hist_equ_12012223(img)
# plt.imshow(out_img, cmap="gray")
# out_img.save("Q3_1_2_img.png")
# plt.show()
# x = range(len(input_hist))
# plt.bar(x, input_hist * 25, color='black', edgecolor='skyblue')
# x_label = ['{}'.format(i) for i in x]
# plt.xticks(x[::50], x_label[::50])
# plt.xlabel('Intensity')
# plt.ylabel('Number of pixels(X $10^{4}$)')
# plt.title('Histogram of the Input Image')
# plt.savefig('Q1_2_in.png')
# plt.show()
#
# plt.bar(x, out_hist * 25, color='black', edgecolor='skyblue')
# x_label = ['{}'.format(i) for i in x]
# plt.xticks(x[::50], x_label[::50])
# plt.xlabel('Intensity')
# plt.ylabel('Number of pixels(X $10^{4}$)')
# plt.title('Histogram of the Output Image')
# plt.savefig('Q1_2_out.png')
# plt.show()
