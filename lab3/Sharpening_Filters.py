import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_blur(image, kernel_size, sigma):
    '''
    高斯模糊函数
    :param image: 输入图像，为一个512*512的nparray
    :param kernel_size: 卷积核大小，一般为奇数，如3、5、7等
    :param sigma: 高斯核标准差
    :return: 输出图像，为一个512*512的nparray
    '''
    # 利用OpenCV库中的GaussianBlur函数实现高斯模糊
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    return blurred_image


def unsharp_masking(image, kernel_size, alpha):
    '''
    Unsharp Masking函数
    :param image: 输入图像，为一个512*512的nparray
    :param kernel_size: 卷积核大小，一般为奇数，如3、5、7等
    :param alpha: 锐化强度系数，一般取0.2~0.3之间
    :return: 输出图像，为一个512*512的nparray
    '''
    # 先进行高斯模糊
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # 计算差分图像
    diff_image = image - blurred_image

    # 对差分图像进行加权叠加
    sharpened_image = cv2.addWeighted(image, 1 + alpha, diff_image, -alpha, 0)

    return sharpened_image


def highboost_filtering(image, kernel_size, alpha):
    '''
    Highboost Filtering函数
    :param image: 输入图像，为一个512*512的nparray
    :param kernel_size: 卷积核大小，一般为奇数，如3、5、7等
    :param alpha: 锐化强度系数，一般取0.2~0.3之间
    :return: 输出图像，为一个512*512的nparray
    '''
    # 先进行高斯模糊
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # 计算差分图像
    diff_image = image - blurred_image

    # 对差分图像进行加权叠加
    sharpened_image = cv2.addWeighted(image, alpha + 1, diff_image, alpha, 0)

    return sharpened_image


def laplace_sharpening(img):
    # 定义Laplace算子
    laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    # 进行卷积
    laplacian = cv2.filter2D(img, -1, laplacian_kernel)
    # 将图像与Laplace图像相加
    sharpened = cv2.addWeighted(img, 1.0, laplacian, -0.5, 0)
    return sharpened


def sobel_sharpening(img):
    # 计算x方向和y方向的Sobel算子
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # 计算梯度幅值
    grad = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)

    # 对灰度图进行高斯模糊
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 将模糊图与边缘图相减，得到锐化后的图像
    sharpened = cv2.addWeighted(blur, 0.7, grad, 0.3, 0)
    return sharpened


def pipline_image1(img):
    # image1 = cv2.medianBlur(img, 5)
    image1 = laplace_sharpening(img)
    image2 = unsharp_masking(img, 3, 0.1)
    image3 = gaussian_blur(image2, 3, 0.3)
    image3 = cv2.medianBlur(image3, 5)
    return (image1, image2, image3)


def pipline_image2(img):
    image1 = cv2.medianBlur(img, 3)
    image1 = unsharp_masking(image1, 3, 0.1)
    # image1 = laplace_sharpening(image1)
    # image1=sobel_sharpening(image1)
    image2 = highboost_filtering(image1, 3, 0.1)
    # image3 = cv2.medianBlur(image2, 3)

    # image3 = match.hist_match_12012223(image2, match.array_to_hist(img))
    image3 = laplace_sharpening(image2)
    # image4 = interpolate.bicubic_yujincheng(image3,[round(image3.shape[0] * 0.9), round(image3.shape[1] * 0.9)])
    # image4 = interpolate.bicubic_yujincheng(image4,[round(image3.shape[0]), round(image3.shape[1])])

    # image3 = cv2.blur(image3, (5, 5))

    # image3 = gaussian_blur(image3, 3, 0.1)
    # image3 = sobel_sharpening(image3)
    # image3 = laplace_sharpening(image3)
    return (image1, image2, image3)


img0 = 'E:/Code/DIP/lab2/Image/9.jpg'
img1 = 'Q4_2.tif'

f = cv2.imread(img0, cv2.IMREAD_GRAYSCALE)
g = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
# kernel1 = np.asarray([[0, 1, 0],
#                       [1, -4, 1],
#                       [0, 1, 0]])
#
# kernel2 = np.asarray([[1, 1, 1],
#                       [1, -8, 1],
#                       [1, 1, 1]])
path = "E:/Code/DIP/lab2/Image_equ/diff_equ"
save_path="E:/Code/DIP/lab2/Image_equ/test"


for filename in os.listdir(path):
    # 检查文件是否是图片文件
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = cv2.imread(os.path.join(path, filename))
        img_sober, img_med, img_out = pipline_image1(img)
        new_filename = filename.split('.')[0] + '_diff_equ.jpg'
        cv2.imwrite(os.path.join(save_path, new_filename), img_med)
# img_sober, img_med, img_out = pipline_image1(f)
# plt.figure(figsize=(10, 14))
# plt.subplot(221)
# plt.imshow(f, cmap='gray')
# plt.title('Origin')
# plt.subplot(222)
# plt.imshow(img_sober, cmap='gray')
# plt.title('image1')
# plt.subplot(223)
# plt.imshow(img_med, cmap='gray')
# plt.title('image2')
# plt.subplot(224)
# plt.imshow(img_out, cmap='gray')
# plt.title('image3')
# plt.show()
# cv2.imwrite('output_1.png', img_out)
#
# image1, image2, image3 = pipline_image2(g)
# plt.figure(figsize=(10, 14))
# plt.subplot(221)
# plt.imshow(g, cmap='gray')
# plt.title('Origin')
# plt.subplot(222)
# plt.imshow(image1, cmap='gray')
# plt.title('image1')
# plt.subplot(223)
# plt.imshow(image2, cmap='gray')
# plt.title('image2')
# plt.subplot(224)
# plt.imshow(image3, cmap='gray')
# plt.title('image3')
# plt.show()
# cv2.imwrite('output_2.png', image3)
