import cv2 as cv
import numpy as np
import os
from PIL import Image, ImageEnhance,ImageOps, ImageChops
import matplotlib.pyplot as plt
import warnings
import time
import function_喻锦城_12012223.hist_equ_12012223 as he
import function_喻锦城_12012223.hist_local_equ_12012223 as hle
path = "Image/"
save_path="Image_equ/test"


# 读取图像
img_sum = []

# 取平均
# img_avg = img_sum.point(lambda i: i/3)

# 显示结果
# img_avg.show()
# 降低曝光度

for filename in os.listdir(path):
    # 检查文件是否是图片文件
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # 打开图片文件
        img = Image.open(os.path.join(path, filename))
        enhancer = ImageEnhance.Brightness(img)
        img_adjusted = enhancer.enhance(0.5)
        if filename.split('.')[0]=="1":
            img_sum = img_adjusted
        else:
            img_sum = Image.blend(img_sum, img_adjusted, 1/25)
        # out_img, out_hist, input_hist = he.hist_equ_12012223(img)
        # out_img_l, out_hist_l, input_hist_l = hle.hist_local_equ_12012223(img, 3)

        # img_inverted = ImageOps.invert(img)

        # 显示图像

        # plt.imshow(out_img, cmap="gray")
        # out_img.save("Image_equ/Img1_equ.png")
        # plt.show()
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
        # new_filename = filename.split('.')[0] + '_equ.jpg'
        # out_img.save(os.path.join(save_path, new_filename))
        # new_filename_local = filename.split('.')[0] + '_local_3.jpg'
        # out_img_l.save(os.path.join(save_path, new_filename_local))
        # new_filename_adj = filename.split('.')[0] + '_adj.jpg'
        # img_adjusted.save(os.path.join(save_path, new_filename_adj))
        # new_filename_inv = filename.split('.')[0] + '_inv.jpg'
        # img_inverted.save(os.path.join(save_path, new_filename_inv))
# img_avg = img_sum.point(lambda i: i/25)
img_sum.show()
for filename in os.listdir(path):
    # 检查文件是否是图片文件
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # 打开图片文件
        img = Image.open(os.path.join(path, filename))
        enhancer = ImageEnhance.Brightness(img)
        img_adjusted = enhancer.enhance(0.5)
        img_diff = ImageChops.subtract(img_adjusted, img_sum)
        # new_filename_sub = filename.split('.')[0] + '_diff.jpg'
        # img_diff.save(os.path.join(save_path, new_filename_sub))
        # 读取图像并将其转换为灰度图像


        # 将图像转换为 NumPy 数组
        new_filename = filename.split('.')[0] + '_diff_equ.jpg'
        img_diff.save(os.path.join(save_path, new_filename))



