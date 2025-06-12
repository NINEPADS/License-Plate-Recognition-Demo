import cv2
import numpy as np
import onnxruntime as ort
import os

class Detector:
    def __init__(self, model_path, confidence_thres=0.5, iou_thres=0.45):
        self.model_path = model_path
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = {0: 'license_plate'}  # 根据你的数据集修改类别
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.session = self.load_model()

    def load_model(self):
        # 加载ONNX模型
        return ort.InferenceSession(self.model_path)

    def preprocess(self, img):
        # 获取输入图像的高度和宽度
        img_height, img_width = img.shape[:2]
        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
        img, ratio, (dw, dh) = self.letterbox(img, new_shape=(640, 640))
        print(f"img shape: {img.shape}")
        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0
        # 将图像的通道维度移到第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先
        # 扩展图像数据的维度，以匹配模型输入的形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data, img_height, img_width, ratio, dw, dh
  
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 向下和向上取整，确保总填充像素正确
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        # 确保最终图像尺寸为 new_shape
        img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        
        return img, (r, r), (dw, dh)

    def postprocess(self, img, output, img_height, img_width, ratio, dw, dh):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                x -= dw
                y -= dh
                x /= ratio[0]
                y /= ratio[1]
                w /= ratio[0]
                h /= ratio[1]
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        valid_boxes = []
        valid_scores = []
        valid_class_ids = []
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            valid_boxes.append(box)
            valid_scores.append(score)
            valid_class_ids.append(class_id)
        return valid_boxes, valid_scores, valid_class_ids

    def infer(self, img):
        image_data, img_height, img_width, ratio, dw, dh = self.preprocess(img)
        print(image_data.shape)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: image_data})
        boxes, scores, class_ids = self.postprocess(img, outputs, img_height, img_width, ratio, dw, dh)
        return boxes, scores, class_ids

    def draw_detections(self, img, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, w, h = box
            color = self.color_palette[class_id]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            label = f"{self.classes[class_id]}: {score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
            cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def save_cropped_images(self, img, boxes, scores, class_ids, output_dir="./data/cropped_images"):
        os.makedirs(output_dir, exist_ok=True)
        img_height, img_width = img.shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, w, h = box
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x1 + w)
            y2 = min(img_height, y1 + h)
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
            save_path = os.path.join(output_dir, f"crop_{i}.jpg")
            cv2.imwrite(save_path, cropped_img)
            print(f"Saved cropped image: {save_path}")
            return save_path
        

    def inference(self, img_path, output_dir="./data/cropped_images"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return None

        boxes, scores, class_ids = self.infer(img)
        output_path = self.save_cropped_images(img, boxes, scores, class_ids, output_dir)

        # 绘制检测框并保存处理后的图片
        # output_path = self.draw_detections(img, boxes, scores, class_ids)
        # output_path = os.path.join(output_dir, "detection_result.jpg")
        # cv2.imwrite(output_path, img)
        print(f"Saved detection result: {output_path}")
        

        return output_path
if __name__ == '__main__':
    detector = Detector(model_path="../model/Detector.onnx")
    output_path = detector.inference(img_path="../data/R.jpg")
    print(f"处理后的图片保存路径: {output_path}")