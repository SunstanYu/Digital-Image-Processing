import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def notch_filter(img):
    rows, cols = img.shape
    P = 2 * rows
    Q = 2 * cols
    H_NF = np.ones((P, Q), np.float32)

    # Define notch pairs coordinates
    notch_pairs = [
        (55, 77),
        (159, 59),
        (84, -54),
        (167, -54)
    ]

    for v_k, u_k in notch_pairs:
        # Calculate notch frequency
        D = 30  # radius of cutoff frequency
        D_k = np.sqrt((np.arange(P)[:, np.newaxis] - u_k) ** 2 + (np.arange(Q) - v_k) ** 2)
        D_k[D_k == 0] = 1  # Avoid division by zero
        # Apply notch filter
        H_NF *= 1 / (1 + (D / D_k) ** 8)

    # Apply filter
    F = np.fft.fft2(img)
    G = np.fft.fftshift(F) * H_NF
    g = np.fft.ifft2(np.fft.ifftshift(G)).real

    # Normalize and convert to uint8
    g *= 255 / np.max(g)
    g = np.uint8(g)

    return g


def output_adjust(output_image):
    output_image *= (output_image > 0)
    output_image = output_image * (output_image <= 255) + 255 * (output_image > 255)
    output_image = output_image.astype(np.uint8)
    return output_image

# Load input image
in_img = cv.imread('Q5_3.tif')
img = in_img[:, :, 0].astype(np.float32)

# Perform notch filtering
mask = notch_filter(img)
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
spectrum_after_filtering = 20 * np.log(np.abs(dst_freq_shift))
spec_out = dst_freq_shift
dst_shift = np.fft.ifft2(dst_freq_shift).real
for i in range(P):
    for j in range(Q):
        dst_shift[i, j] *= (-1) ** (i + j)  # ifftshift

# Crop to original size and adjust pixel values
dst = dst_shift[:height, :width]
dst *= 255 / np.max(dst)
dst = np.uint8(dst)

# Show and save images
original_spectrum = 10 * np.log(np.abs(spec_in))
cv.imshow('Original Spectrum', original_spectrum.astype(np.uint8))
cv.imshow('Filter Mask', mask)
cv.imshow('Filtered Spectrum', spectrum_after_filtering.astype(np.uint8))
cv.imshow('Output Image', dst)
cv.waitKey(0)
cv.destroyAllWindows()