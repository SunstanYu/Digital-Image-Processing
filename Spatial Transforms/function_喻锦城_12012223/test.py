import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('all_out_pure.png', cv2.IMREAD_GRAYSCALE)

brightness_factor = 0.9 # 增加亮度，可设置为小于1的值来降低亮度

# 调整亮度
brighter_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
kernel_size = (3, 3)

# 对图像进行高斯模糊
blurred_image = cv2.GaussianBlur(brighter_image, kernel_size, sigmaX=0.5)
# 显示原始图像和调整后的图像
cv2.imshow("Original Image", image)
cv2.imshow("Brighter Image", blurred_image)
cv2.imwrite("all_dark.png",blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#
# phone_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
# plt.hist(image.ravel(), bins=256)  # numpy中的ravel将数组多维度拉成一维数组
# plt.show()
#
# phone_equalize = cv2.equalizeHist(image)
# plt.hist(phone_equalize.ravel(), bins=256)  # numpy中的ravel将数组多维度拉成一维数组
# plt.show()
#
# res = np.hstack((image, phone_equalize))  # 横向拼接,将多个数组按水平方向(列顺序)堆叠成一个新的数组。
# cv2.imshow('phone_equalize', res)
# cv2.waitKey(100000)
# clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(4, 4))  # 通过类创建了一个局部均衡化对象
# phone_clahe = clahe.apply(image)
# res = np.hstack((image, phone_equalize, phone_clahe))
# cv2.imshow('phone_equalize1', phone_clahe)
# # cv2.imwrite("dar_img_out.png",phone_clahe)
# cv2.waitKey(0)
# cv2.destroyAllWindows()