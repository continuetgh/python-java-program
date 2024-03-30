from pathlib import Path

import torch

# 初始化模型
model = torch.hub.load('.', 'custom', path='runs/train/exp13/weights/best.pt', source='local')

# 如果需要GPU支持
if torch.cuda.is_available():
    model.to('cuda')

# 读取图片所在的目录，此目录可以为系统本地路径，也可以是微信小程序上传路径，更加可以是服务器路径
img_dir = Path("D:/Yolov5/testyolov5/data/mydata/images")

# 假设你知道类别的索引与名称的映射
# 例如：'0': 'person', '1': 'bicycle', '2': 'car', '3': 'parking', ...
# 你需要根据你的模型训练时的类别映射来设置这个字典
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

#开始写文件
result_file = open("parkingresults2.txt", "w")

# 遍历目录中的每个图片文件
for img_file in img_dir.glob('*.jpg'):  # 假设图片是jpg格式
    img_path = str(img_file)
    # Inference
    results = model(img_path, augment=False)  # 假设模型可以直接处理单个图片路径

    # 提取边界框信息和置信度
    vehicle_boxes = []
    parking_boxes = []
    for xyxy, conf, cls in zip(results.xyxy[0], results.xyxy[0][:, 4], results.xyxy[0][:, 5]):  # 注意索引[0]可能需要根据实际情况调整
        if conf.item() > 0.5:  # 过滤掉置信度较低的检测结果
            cls_name = LABELS[int(cls.item())]  # 将cls.item()转换为整数
            if cls_name in ['bike', 'car', 'van']:
                vehicle_boxes.append(xyxy)
            elif cls_name == 'parking':
                parking_boxes.append(xyxy)

    # 遍历每个车辆和每个车位
    for vehicle_box in vehicle_boxes:
        parked_properly = False  # 假设车辆没有正确停放
        for parking_box in parking_boxes:
            iou = calculate_iou(vehicle_box, parking_box)
            if iou > 0.3:  # 根据实际情况调整IoU阈值
                parked_properly = True
                break  # 一旦找到匹配的车位，跳出循环
        if not parked_properly:
            result_message = f"{img_file.name}: 未正确停放\n"
            print(result_message)
            result_file.write(result_message)
        else:
            result_message = f"{img_file.name}: 正确停放\n"
            print(result_message)
            result_file.write(result_message)

#关闭文件
result_file.close()
