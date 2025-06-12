import cv2
import numpy as np
from rknnlite.api import RKNNLite

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
        ]

class LicensePlateRecognizer:
    def __init__(self, model_path):
        # 初始化 RKNNLite
        self.rknn_lite = RKNNLite(verbose=False)
        
        # 加载模型
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            raise Exception(f'Load RKNN model "{model_path}" failed!')
            
        print('Model loaded successfully')
        
        # 初始化运行环境
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            raise Exception('Init runtime environment failed!')
            
        print('Runtime initialized successfully')
    
    def decode(self, preds):
        # 解码函数
        pred_labels = list()
        labels = list()
        for i in range(preds.shape[0]):
            pred = preds[i, :, :]
            pred_label = list()
            for j in range(pred.shape[1]):
                pred_label.append(np.argmax(pred[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = pred_label[0]
            for c in pred_label:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            pred_labels.append(no_repeat_blank_label)
        
        for i, label in enumerate(pred_labels):
            lb = ""
            for i in label:
                lb += CHARS[i]
            labels.append(lb)
        return labels
    
    def preprocess_image(self, img_path):
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            raise Exception(f'Failed to read image: {img_path}')
            
        # 预处理图片
        img = cv2.resize(img, (94, 24))
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def inference(self, img_path):
        try:
            # 图片预处理
            img = self.preprocess_image(img_path)
            
            # 模型推理
            outputs = self.rknn_lite.inference(inputs=[img])
            
            # 后处理
            labels = self.decode(outputs[0])
            
            return labels[0]
        
        finally:
            # 释放资源
            self.rknn_lite.release()

# 使用示例
if __name__ == '__main__':
    recognizer = LicensePlateRecognizer(model_path='../model/lprnet.rknn')
    result = recognizer.inference('../model/test.jpg')
    print('车牌识别结果:', result)