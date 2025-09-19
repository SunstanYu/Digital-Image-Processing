import cv2
import numpy as np

import cv2

def sobel_sharpen(input_image):
    sobel_calx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # generate spatial sobel x
    sobel_caly = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # generate spatial sobel y
    gradient_imgx = cv2.Sobel(input_image, cv2.CV_32F, 1, 0, ksize=3)  # sobel filter along x-axis
    gradient_imgy = cv2.Sobel(input_image, cv2.CV_32F, 0, 1, ksize=3)  # sobel filter along y-axis
    gradient_img = cv2.convertScaleAbs(cv2.addWeighted(gradient_imgx, 0.5, gradient_imgy, 0.5, 0))  # combine two sobel filters
    output_image = cv2.add(input_image, gradient_img)  # add gradient to input
    output_image = output_adjust(output_image)
    return output_image


def output_adjust(output_image):
    output_image *= (output_image > 0)
    output_image = output_image * (output_image <= 255) + 255 * (output_image > 255)
    output_image = output_image.astype(np.uint8)
    return output_image


img = cv2.imread('Q5_1.tif', cv2.IMREAD_GRAYSCALE)

dst = sobel_sharpen(img)
cv2.imwrite('spat_output_Q5_1_12012223.png', dst)

cv2.imshow('output', dst)
cv2.waitKey(0)