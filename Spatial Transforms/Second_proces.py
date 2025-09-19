import os
from PIL import Image, ImageEnhance,ImageOps, ImageChops
import function_喻锦城_12012223.reduce_SAP_12012223 as sap

path = "Image_equ/diff_equ"
save_path="Image_equ/test"


for filename in os.listdir(path):
    # 检查文件是否是图片文件
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = Image.open(os.path.join(path, filename))
        output_median_image, output_mean_image=sap.reduce_SAP_12012223(img, 3)
        new_filename = filename.split('.')[0] + '_diff_equ.jpg'
        output_median_image.save(os.path.join(save_path, new_filename))
