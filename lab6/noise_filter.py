import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2


def spectrum_show(img):
    """
    :param img: input image
    :return:
    """
    f = np.fft.fft2(img)  # 快速傅里叶变换算法得到频率分布
    fshift = np.fft.fftshift(f)  # 将图像中的低频部分移动到图像的中心，默认是在左上角
    fimg = np.log(np.abs(fshift))  # fft结果是复数, 其绝对值结果是振幅，取对数的目的是将数据变换到0~255
    hist = array_to_hist(fimg)
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Fourier')
    plt.axis('off')
    plt.subplot(132), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
    plt.axis('off')
    plt.subplot(133)
    x = range(len(hist))
    plt.bar(x, hist * 25, color='black', edgecolor='skyblue')
    x_label = ['{}'.format(i) for i in x]
    plt.xticks(x[::50], x_label[::50])
    plt.xlabel('Intensity')
    plt.ylabel('Number of pixels(X $10^{4}$)')
    plt.title('Histogram of the Input Image')
    # plt.savefig('Q1_1_in.png')
    plt.show()


def show_img(img):
    plt.imshow(img, 'gray')
    plt.title('Original Fourier')
    plt.show()


def array_to_hist(arr):
    """
    :param arr: input image
    :return: corresponding histogram
    """
    # arr = np.array(img).astype(np.int32)
    arr = np.int32(arr)
    [width, height] = arr.shape
    output_hist = np.zeros(256)
    for line in arr:
        for point in line:
            output_hist[point] += 1
    N = width * height
    output_hist = output_hist / N
    return output_hist


def harmonic(arr, m, n):
    """
    :param arr:
    :param m: row num
    :param n: column num
    :return: output1 array
    """
    height, width = arr.shape
    rows = height // m
    cols = width // n
    arr_mean = np.mean(arr)
    # 将所有值为0的像素替换为邻域内像素的平均值
    arr[arr == 0] = arr_mean
    for i in range(rows):
        for j in range(cols):
            y = i * m
            x = j * n
            seg = arr[y:y + m, x:x + n]
            arr[y, x] = m * n / harmonic_sum_2d(seg)
    return arr


def contraharmonic(arr, size, q):

    dst = arr.copy()
    height, width = arr.shape

    half_size = int((size - 1) / 2)
    for i in range(half_size, height - half_size):
        for j in range(half_size, width - half_size):
            seg = arr[i - half_size:i + half_size, j - half_size:j + half_size]

            numerator = harmonic_sum_q(seg, q + 1)
            dominator = harmonic_sum_q(seg, q)
            if dominator == 0:
                dst[i, j] = 0
            else:
                dst[i, j] = numerator / dominator

    return dst


def harmonic_sum_2d(arr):
    total = 0
    for row in arr:
        for val in row:
            total += 1 / val
    return total


def harmonic_sum_q(arr, q):
    total = 0
    for row in arr:
        for val in row:
            if val == 0 and q < 0:
                return 0
            else:
                total += val ** q
    return total


def order_statistic(arr, size, mode):
    height, width = arr.shape
    dst = np.zeros((height, width), np.uint16)

    # arr_mean = np.mean(arr)
    # 将所有值为0的像素替换为邻域内像素的平均值
    # arr[arr == 0] = 1
    half_size = int((size - 1) / 2)
    for i in range(half_size, height - half_size):
        for j in range(half_size, width - half_size):
            seg = arr[i - half_size:i + half_size, j - half_size:j + half_size]

            if mode == "max":
                dst[i, j] = np.max(seg)
            elif mode == "min":
                dst[i, j] = np.min(seg)
            elif mode == "median":
                dst[i, j] = np.median(seg)
            elif mode == "mean":
                dst[i, j] = np.mean(seg)
    return dst


def gaussian_filter(img, kernel_size=3, sigma=1):
    """
    使用高斯滤波器对图像进行平滑处理
    :param img: 待处理的图像，要求为灰度图像
    :param kernel_size: 滤波器的大小，要求为奇数
    :param sigma: 高斯核的标准差
    :return: 处理后的图像
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    dst = cv2.filter2D(img, -1, kernel * kernel.T)
    return dst



def adaptive_median(arr, window_size=3, max_window_size=7):
    rows, cols = arr.shape
    dst = np.zeros((rows, cols), np.uint16)
    for i in range(rows):
        for j in range(cols):
            window_size_now = window_size
            while window_size_now <= max_window_size:
                window = arr[max(0, i - window_size_now // 2):min(rows, i + window_size_now // 2 + 1),
                         max(0, j - window_size_now // 2):min(cols, j + window_size_now // 2 + 1)]
                median = np.median(window)
                maximum = np.max(window)
                minimum = np.min(window)
                if median - minimum > 0 and median - maximum < 0:
                    if arr[i, j] - minimum > 0 and arr[i, j] - maximum < 0:
                        dst[i, j] = arr[i, j]
                        break
                    else:
                        dst[i, j] = median
                        break
                else:
                    window_size_now += 2
    return dst


if __name__ == '__main__':
    # frequency domain
    # path_load = "resource"
    # for filename in os.listdir(path_load):
    #     if filename.startswith('Q6_1'):
    #         img = Image.open(os.path.join(path_load, filename))
    #
    #         spectrum_show(img)
    #
    # #image 1
    img11 = Image.open("resource/Q6_1_1.tiff")
    input_arr = np.array(img11)
    output1 = contraharmonic(input_arr, 3, 1.5)
    # out = Image.fromarray(output1)
    # out.save("Q1_1_out.png")
    plt.subplot(121)
    plt.imshow(img11, 'gray')
    plt.title('Original')
    plt.subplot(122)
    plt.imshow(output1, 'gray')
    plt.title('after')
    plt.show()
    #
    # #image 2
    img12 = Image.open("resource/Q6_1_2.tiff")
    input_arr = np.array(img12)
    output1 = contraharmonic(input_arr, 3, -1.5)
    # out = Image.fromarray(output1)
    # out.save("Q1_2_out.png")
    plt.subplot(121)
    plt.imshow(img12, 'gray')
    plt.title('Original')
    plt.subplot(122)
    plt.imshow(output1, 'gray')
    plt.title('after')
    plt.show()

    # image 3
    img13 = Image.open("resource/Q6_1_3.tiff")
    input_arr = np.array(img13)
    output1 = adaptive_median(input_arr)
    # out = Image.fromarray(output1).convert("L")
    # out.save("Q1_3_out.png")
    plt.subplot(121)
    plt.imshow(img13, 'gray')
    plt.title('Original')
    plt.subplot(122)
    plt.imshow(output1, 'gray')
    plt.title('after')
    plt.show()

    # image4
    img14 = Image.open("resource/Q6_1_4.tiff")
    input_arr = np.array(img14)
    output1 = adaptive_median(input_arr)
    output2 = gaussian_filter(output1)
    # out = Image.fromarray(output1).convert("L")
    # out.save("Q1_4_out_mid.png")
    plt.subplot(131)
    plt.imshow(img14, 'gray')
    plt.title('Original')
    plt.subplot(132)
    plt.imshow(output1, 'gray')
    plt.title('intermediate')
    plt.subplot(133)
    plt.imshow(output2, 'gray')
    plt.title('final')
    plt.show()
