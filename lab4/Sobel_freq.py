import cv2 as cv
import numpy as np


def sobel_freq(input_image):
    height, width = input_image.shape
    pad_image = np.zeros((height + 2, width + 2))  # zero padding
    pad_image[:height, :width] = input_image
    # for i in range(height + 2):
    #     for j in range(width + 2):
    #         pad_image[i, j] *= (-1) ** (i + j)
    img_freq = np.fft.fft2(pad_image)
    sobel_calx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)  # generate spatial sobel x
    sobel_caly = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # generate spatial soebl y

    edge_x = cv.filter2D(np.real(np.fft.ifft2(img_freq)), -1, sobel_calx)
    edge_y = cv.filter2D(np.real(np.fft.ifft2(img_freq)), -1, sobel_caly)
    edge = np.abs(edge_x) + np.abs(edge_y)  # get edge
    edge_de = edge[:height, :width]
    output_image = edge_de + input_image  # adding edge and input image
    output_image = output_adjust(output_image)
    return output_image


def output_adjust(output_image):
    output_image *= (output_image > 0)
    output_image = output_image * (output_image <= 255) + 255 * (output_image > 255)
    output_image = output_image.astype(np.uint8)
    return output_image


img = cv.imread('Q5_1.tif', cv.IMREAD_GRAYSCALE)

dst = sobel_freq(img)
# cv.imwrite('freq_output_Q5_1_12012223.png', dst)

cv.imshow('output', dst)
cv.waitKey(0)