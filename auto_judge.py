from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import torch
import json
import numpy as np

app = Flask(__name__)
# 创建空列表以存储结果
results_list = []
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
    for xyxy, conf, cls in zip(results.xyxy[0], results.xyxy[0][:, 4], results.xyxy[0][:, 5]):
        if conf.item() > 0.5:
            cls_name = LABELS[int(cls.item())]
            if cls_name in ['bike', 'car', 'van']:
                vehicle_boxes.append(xyxy)
            elif cls_name == 'parking':
                parking_boxes.append(xyxy)

    # 检查车辆停放情况
    parked_properly = False
    for vehicle_box in vehicle_boxes:
        for parking_box in parking_boxes:
            iou = calculate_iou(vehicle_box, parking_box)
            if iou > 0.3:
                parked_properly = True
                break
        if parked_properly:
            break

    # 构建结果消息
    if parked_properly:
        result_message = { "file_name": photo.name,"status": "Correctly parked"}
    else:
        result_message = { "file_name": photo.name,"status": "Improperly parked"}

    # 将结果消息添加到结果列表中
    results_list.append(result_message)

    # 将结果列表转换为JSON格式
    result_json = json.dumps(results_list)

    # 返回JSON数据给前端
    return result_json


if __name__ == '__main__':
    app.run(debug=True)