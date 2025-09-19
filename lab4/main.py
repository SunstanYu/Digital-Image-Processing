import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time


def butterworth_notch(img):
    rows, cols = img.shape
    H_NF = np.ones((2 * rows, 2 * cols), np.float32)
    for y in range(-rows, rows):
        for x in range(-cols, cols):
            D = 30  # we define D0=30 for radius(cutoff freq)
            v_k = 55  # the coordinate of the 1st notch pair
            u_k = 77
            D_k = ((y + u_k) ** 2 + (x + v_k) ** 2) ** 0.5
            if D_k != 0:  # to avoid divided by 0
                H_NF[y + rows, x + cols] = H_NF[y + rows, x + cols] * 1 / (1 + (D / D_k) ** 8)
            else:
                H_NF[y + rows, x + cols] = 0
            D_k = ((y - u_k) ** 2 + (x - v_k) ** 2) ** 0.5
            if D_k != 0:
                H_NF[y + rows, x + cols] = H_NF[y + rows, x + cols] * 1 / (1 + (D / D_k) ** 8)
            else:
                H_NF[y + rows, x + cols] = 0
            v_k = 55  # the coordinate of the 2nd notch pair
            u_k = 150
            D_k = ((y + u_k) ** 2 + (x + v_k) ** 2) ** 0.5
            if D_k != 0:
                H_NF[y + rows, x + cols] = H_NF[y + rows, x + cols] * 1 / (1 + (D / D_k) ** 8)
            else:
                H_NF[y + rows, x + cols] = 0
            D_k = ((y - u_k) ** 2 + (x - v_k) ** 2) ** 0.5
            if D_k != 0:
                H_NF[y + rows, x + cols] = H_NF[y + rows, x + cols] * 1 / (1 + (D / D_k) ** 8)
            else:
                H_NF[y + rows, x + cols] = 0
            v_k = -55  # the coordinate of the 3rd notch pair
            u_k = 86
            D_k = ((y + u_k) ** 2 + (x + v_k) ** 2) ** 0.5
            if D_k != 0:
                H_NF[y + rows, x + cols] = H_NF[y + rows, x + cols] * 1 / (1 + (D / D_k) ** 8)
            else:
                H_NF[y + rows, x + cols] = 0
            D_k = ((y - u_k) ** 2 + (x - v_k) ** 2) ** 0.5
            if D_k != 0:
                H_NF[y + rows, x + cols] = H_NF[y + rows, x + cols] * 1 / (1 + (D / D_k) ** 8)
            else:
                H_NF[y + rows, x + cols] = 0
            v_k = -55  # the coordinate of the 4th notch pair
            u_k = 150
            D_k = ((y + u_k) ** 2 + (x + v_k) ** 2) ** 0.5
            if D_k != 0:
                H_NF[y + rows, x + cols] = H_NF[y + rows, x + cols] * 1 / (1 + (D / D_k) ** 8)
            else:
                H_NF[y + rows, x + cols] = 0
            D_k = ((y - u_k) ** 2 + (x - v_k) ** 2) ** 0.5
            if D_k != 0:
                H_NF[y + rows, x + cols] = H_NF[y + rows, x + cols] * 1 / (1 + (D / D_k) ** 8)
            else:
                H_NF[y + rows, x + cols] = 0
    return H_NF


def output_adjust(output_image):
    output_image *= (output_image > 0)
    output_image = output_image * (output_image <= 255) + 255 * (output_image > 255)
    output_image = output_image.astype(np.uint8)
    return output_image


time_start = time.time()
in_img = cv.imread('E:/Code/DIP/lab2/Image/1.jpg')
img = in_img[:, :, 0].astype(np.float32)
mask = butterworth_notch(img)
height, width = img.shape
P = 2 * height
Q = 2 * width
img_large = np.zeros((P, Q), dtype=np.float32)
for i in range(height):  # fftshift
    for j in range(width):
        img[i, j] *= (-1) ** (i + j)
img_large[:height, :width] = img  # enlarge
img_freq = np.fft.fft2(img_large)  # fft
spec_in = img_freq
dst_freq_shift = img_freq * mask  # operation
dst_shift = np.fft.ifft2(dst_freq_shift)  # ifft
dst_shift = np.real(dst_shift)
for i in range(P):
    for j in range(Q):
        dst_shift[i, j] *= (-1) ** (i + j)  # ifftshift
dst = dst_shift[:height, :width]

time_end = time.time()
print('time cost', time_end - time_start, 's')

mask_spectrum = 20 * np.log(np.abs(mask))
# cv.imwrite('filter_Q5_3_12012223.png', mask)
# cv.imwrite('output_Q5_3_12012223.png', dst)
original_spectrum = 10 * np.log(np.abs(spec_in))
plt.imshow(original_spectrum, cmap='gray')
spec_out = dst_freq_shift
spectrum_after_filtering = 10 * np.log(np.abs(spec_out))
plt.imshow(spectrum_after_filtering, cmap='gray')
# cv.imwrite('input_spectrum_Q5_3_12012223.png', original_spectrum)
# cv.imwrite('output_spectrum_Q5_3_12012223.png', spectrum_after_filtering)
dst = output_adjust(dst)
spectrum_after_filtering = output_adjust(spectrum_after_filtering)
cv.imshow('input_spec', original_spectrum.astype(np.uint8))
cv.imshow('filter_spec', mask)
cv.imshow('output_image', dst)
cv.imshow('output_spec', spectrum_after_filtering)
cv.waitKey(0)
