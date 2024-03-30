from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import torch
import json
import numpy as np

app = Flask(__name__)

# 初始化模型
model = torch.hub.load('.', 'custom', path='runs/train/exp13/weights/best.pt', source='local')
LABELS = {0: 'bike', 1: 'car', 2: 'van', 3: 'parking'}  # 替换为实际的映射

# 是否违规停车地判断逻辑
def calculate_iou(boxA, boxB):
    xA1, yA1, xA2, yA2, _, _ = boxA
    xB1, yB1, xB2, yB2, _, _ = boxB

    # 计算交集区域
    interArea = max(0, min(xA2, xB2) - max(xA1, xB1)) * max(0, min(yA2, yB2) - max(yA1, yB1))

    # 计算并集区域
    boxAArea = (xA2 - xA1) * (yA2 - yA1)
    boxBArea = (xB2 - xB1) * (yB2 - yB1)
    unionArea = boxAArea + boxBArea - interArea

    # 避免除数为0的情况
    iou = interArea / max(unionArea, 1e-6)
    return iou

@app.route('/detect-parking', methods=['POST'])
def detect_parking():
    # 接收小程序传来的照片数据
    photo = request.files['file']

    # 读取照片并进行模型推断
    img = Image.open(BytesIO(photo.read()))
    results = model(img, augment=False)

    # 提取边界框信息和置信度
    vehicle_boxes = []
    parking_boxes = []

    # 将 Tensor 对象转换为 NumPy 数组
    def tensor_to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # 将结果中的 Tensor 对象转换为 NumPy 数组
    for xyxy, conf, cls in zip(results.xyxy[0], results.xyxy[0][:, 4], results.xyxy[0][:, 5]):
        if conf.item() > 0.5:
            cls_name = LABELS[int(cls.item())]
            if cls_name in ['bike', 'car', 'van']:
                vehicle_boxes.append(tensor_to_numpy(xyxy))  # 转换为 NumPy 数组
            elif cls_name == 'parking':
                parking_boxes.append(tensor_to_numpy(xyxy))  # 转换为 NumPy 数组

    # 遍历每个车辆和每个车位，判断是否正确停放
    # 遍历每个车辆和每个车位，判断是否正确停放
    result = []
    for vehicle_box in vehicle_boxes:
        parked_properly = False
        for parking_box in parking_boxes:
            iou = calculate_iou(vehicle_box, parking_box)
            if iou > 0.3:
                parked_properly = True
                break
        if parked_properly:
            result.append({'vehicle_box': vehicle_box, 'parked_properly': '正确'})
        else:
            result.append({'vehicle_box': vehicle_box, 'parked_properly': '错误'})
    # Assuming 'result' is the NumPy array you're trying to jsonify
    result_list = result.tolist()
    # Now jsonify the list
    return jsonify({'objects': result_list})


if __name__ == '__main__':
    app.run(debug=True)