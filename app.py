from YOLO_Detector.python.detector import Detector
from LPRNet.python.lprnet import LicensePlateRecognizer
import cv2
import sys
import os

def main(input_image_path):
    # 初始化检测器和识别器
    detector = Detector(model_path="YOLO_Detector/model/Detector.onnx")
    recognizer = LicensePlateRecognizer(model_path="LPRNet/model/lprnet.rknn")

    try:
        # 检测车牌并保存处理后的图片
        # 读取检测结果图片路径（这里假设处理后的图片路径由detector.inference返回）
        detection_result_path = detector.inference(input_image_path)
        print(f"detection result path:{detection_result_path}")
        if detection_result_path is None:
            print("No detection result image available.")
            return
        # 调用识别器识别车牌
        recognition_result = recognizer.inference(detection_result_path)
        print(f"License plate content of {detection_result_path}: {recognition_result}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python app.py <input_image_path>")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    main(input_image_path)