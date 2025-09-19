import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def inverse_filter(img):
    rows, cols = img.shape
    center = [int(rows / 2), int(cols / 2)]
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = np.exp(-(-0.0025 * ((i - center[0]) ** 2 + (j - center[1]) ** 2) ** (5 / 6)))
    return mask


def butterworth_filter(img, radius=30, n=2):
    rows, cols = img.shape
    center = [int(rows / 2), int(cols / 2)]
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance_u_v = (i - center[0]) ** 2 + (j - center[1]) ** 2
            mask[i, j] = 1 / (1 + (distance_u_v ** 0.5 / radius) ** (2 * n))
    return mask

def wiener_filter(img, K=0.001):
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            mask[u, v] = np.exp(-0.0025 * ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** (5 / 6))
    mask = np.conj(mask) * mask / (mask * (np.conj(mask) * mask + K))
    return mask

def full_inverse(img):
    img_freq = np.fft.fft2(img)
    img_freq = np.fft.fftshift(img_freq)
    mask = inverse_filter(img)
    dst_freq = img_freq * mask
    dst = np.fft.ifft2(dst_freq)
    dst = np.fft.ifftshift(dst)
    dst = np.real(dst)
    return dst

def limited_inverse(img):
    img_freq = np.fft.fft2(img)
    img_freq = np.fft.fftshift(img_freq)
    mask = inverse_filter(img)
    dst_freq = img_freq * mask
    mask = butterworth_filter(img_freq, 30, 10)
    dst_freq *= mask
    dst = np.fft.ifftshift(dst_freq)
    dst = np.fft.ifft2(dst)
    dst = np.real(dst)
    return dst

def wiener(img):
    img_freq = np.fft.fft2(img)
    img_freq = np.fft.fftshift(img_freq)
    mask = wiener_filter(img,0.1)
    dst_freq = img_freq * mask
    dst = np.fft.ifftshift(dst_freq)
    dst = np.fft.ifft2(dst)
    dst = np.real(dst)
    return dst

if __name__ == '__main__':
    img = Image.open("resource/Q6_2.tif")
    input_arr = np.array(img)
    output1 = full_inverse(input_arr)
    # out = Image.fromarray(output1).convert("L")
    # out.save("Q2_1_out.png")
    output2 = limited_inverse(input_arr)
    # out = Image.fromarray(output2).convert("L")
    # out.save("Q2_2_out_80.png")
    output3 = wiener(input_arr)
    # out = Image.fromarray(output3).convert("L")
    # out.save("Q2_3_out_1.png")
    plt.subplot(221)
    plt.imshow(img, 'gray')
    plt.title('Original')
    plt.subplot(222)
    plt.imshow(output1, 'gray')
    plt.title('full inverse')
    plt.subplot(223)
    plt.imshow(output2, 'gray')
    plt.title('limited inverse')
    plt.subplot(224)
    plt.imshow(output3, 'gray')
    plt.title('wiener')
    plt.show()
