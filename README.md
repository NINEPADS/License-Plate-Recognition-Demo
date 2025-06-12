# License Plate Recognition Demo

## 简介
这是一个车牌识别演示项目，包含两个主要子模块：
1. **YOLO_Detector**：基于YOLO的车牌检测模块，用于在图像中定位车牌区域。
2. **LPRNet**：基于LPRNet的车牌识别模块，用于识别车牌中的字符内容。

## 项目结构

```
.
├── app.py			# 项目入口脚本
├── data
│   └── cropped_images
│       └── crop_0.jpg
├── input.jpg
├── LPRNet			# 车牌识别模块
│   ├── model		# 模型文件
│   │   ├── lprnet.rknn
│   │   └── test.jpg
│   └── python		# Python脚本
│       ├── lprnet.py
│       └── __pycache__
│           └── lprnet.cpython-38.pyc
├── README.md		# 项目说明文档
└── YOLO_Detector	# 车牌检测模块
    ├── model		# 模型文件
    │   └── Detector.onnx
    └── python		# Python脚本
        ├── detector.py
        └── __pycache__
            └── detector.cpython-38.pyc

```



## 运行环境

### 软件环境
- Python 3.8+
- OpenCV
- NumPy
- onnxruntime
- RKNN Toolkit

### 硬件环境
- 支持RKNN模型的RK3568设备

## 使用方法

### 准备工作
1. 确保所有依赖库已安装：
   ```bash
   pip install opencv-python numpy onnxruntime

2. 确保项目目录结构正确，并且所有模型文件已就绪。

### 运行演示

1. 准备一张待识别的车牌图片，并将其放置在项目根目录下或指定路径。

2. 运行主脚本：
   ```bash
   python app.py <input_image_path>
   ```
   其中 `<input_image_path>` 是待识别图片的路径。例如：
   ```bash
   python app.py input.jpg
   ```

3. 脚本将首先调用车牌检测模块对输入图片进行车牌区域检测，并保存检测后的图片到 `data/cropped_images` 目录下。

4. 随后，脚本将调用车牌识别模块对检测到的车牌区域进行字符识别，并输出识别结果。

## 示例

### 输入图片
假设输入图片为 `data/input.jpg`，内容为一张包含车牌的车辆图片。

### 运行命令
```bash
python app.py data/input.jpg
```

### 输出结果
1. 车牌区域的裁剪图片将保存在 `data/cropped_images/crop_0.jpg` 文件中。
2. 车牌识别结果将输出到控制台，例如：
   ```
   License plate content of ./data/cropped_images/crop_0.jpg: 鄂J20988
   ```

## 不足之处

1. 没做到YOLO导出的ONNX模型转RKNN（尝试过但是转换后的模型几乎不可用）
2. RKNN Model Zoo官方的车牌识别模型LPRNet失误率高
3. 没实现检测一张图有多个车牌的情况（尝试过但是有问题）

## 引用内容

1. YOLO模型的用于识别车牌的参数来源于[maliahson/YOLO_Lisencse_Plate_Detector at main](https://huggingface.co/spaces/maliahson/YOLO_Lisencse_Plate_Detector/tree/main)的best.pt

2. LPRNet模型来源于[airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo?tab=readme-ov-file)

## 许可证
[Apache License 2.0](./LICENSE)
