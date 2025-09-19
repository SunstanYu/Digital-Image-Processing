# 数字图像处理实验项目 (Digital Image Processing Labs)

本项目包含了一系列数字图像处理算法的实现，涵盖了图像插值、直方图均衡化、空间变换、滤波、频域滤波和图像恢复等核心内容。

## 项目结构

### Image Interpolation - 图像插值算法
- **最近邻插值** (`nearest_yujincheng.py`)
- **双线性插值** (`bilinear_yujincheng.py`) 
- **双三次插值** (`bicubic_yujincheng.py`)

实现了三种经典的图像插值算法，用于图像的放大和缩小操作。

### Spatial Transforms - 直方图处理与空间变换
- **直方图均衡化** (`hist_equ_12012223.py`)
- **局部直方图均衡化** (`hist_local_equ_12012223.py`)
- **直方图匹配** (`hist_match_12012223.py`)
- **椒盐噪声滤波** (`reduce_SAP_12012223.py`)

包含直方图相关的图像增强算法和空间变换操作。

### Sharpen Filter - 锐化滤波器
- **锐化滤波器实现** (`Sharpening_Filters.py`)

实现了图像锐化算法，用于增强图像的边缘和细节信息。

### Frequency Domain Filter - 频域滤波
- **Sobel频域滤波** (`Sobel_freq.py`)
- **Sobel空域滤波** (`Sobel_spac.py`)
- **巴特沃斯陷波滤波器** (`Butterworth_notch.py`)

实现了频域和空域的滤波算法，包括边缘检测和噪声抑制。

### Image Restoration - 图像恢复
- **噪声滤波** (`noise_filter.py`)
- **图像恢复** (`restore_image.py`)

实现了图像去噪和恢复算法，用于处理各种类型的图像退化。

## 技术栈
- **Python 3.x**
- **OpenCV** - 图像处理
- **NumPy** - 数值计算
- **Matplotlib** - 图像显示
- **SciPy** - 科学计算

## 运行环境
```bash
pip install opencv-python numpy matplotlib scipy
```

## 使用方法
每个实验文件夹都包含相应的Python脚本，可以直接运行：
```bash
python "Image Interpolation/nearest_yujincheng.py"
python "Spatial Transforms/function_喻锦城_12012223/hist_equ_12012223.py"
python "Sharpen Filter/Sharpening_Filters.py"
python "Frequency Domain Filter/Sobel_freq.py"
python "Image Restoration/noise_filter.py"
```

## 实验内容
- **图像插值**: 最近邻、双线性、双三次插值算法
- **直方图处理**: 全局和局部直方图均衡化、直方图匹配
- **空间滤波**: 均值滤波、中值滤波、锐化滤波
- **频域滤波**: FFT变换、频域滤波、陷波滤波
- **图像恢复**: 噪声去除、图像修复

## 作者
喻锦城

## 许可证
本项目仅用于学习和研究目的。
