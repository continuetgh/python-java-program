import torch
import cv2
from PIL import Image
import os
import sys

# 添加YOLOv5本地路径到系统路径
sys.path.append('D:\\Yolov5\\testyolov5\\yolov5')

# 从本地路径导入YOLOv5的相关模块
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.general import scale_boxes

# 初始化设备（GPU或CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型，确保模型被移动到正确的设备上
model = attempt_load('runs/train/exp14/weights/best.pt').to(device)
model.eval()  # 设置为评估模式


def detect_license_plate(image_path, model, device):
    # 读取图像
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).to(device).permute(2, 0, 1).float()  # 转换为tensor并移动到设备
    img_tensor /= 255.0  # 归一化到[0, 1]
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # 如果图像是单张，增加一个batch维度

    # 进行推理
    with torch.no_grad():
        pred = model(img_tensor)[0]  # 获取预测结果

    # 应用非极大值抑制
    pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.5)

    # 将坐标从模型输出空间缩放到原始图像空间
    scale = torch.tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(device)
    # 将坐标从模型输出空间缩放到原始图像空间
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[1:], det[:, :4], img.shape).round()

    # 返回检测结果
    return pred, img


# 图像路径
image_path = 'D:/Yolov5/testyolov5/data/parking/parking010.jpg'

# 进行检测
results, img = detect_license_plate(image_path, model, device)

# 在图像上绘制边界框
for det in results:
    if det is not None and len(det):
        # det[:, :4] 是边界框坐标，det[:, 4] 是置信度，det[:, 5:] 是类别ID和分数
        c1, c2 = (int(det[0, 0]), int(det[0, 1])), (int(det[0, 2]), int(det[0, 3]))
        cv2.rectangle(img, c1, c2, (0, 255, 0), 2)

    # 显示图像
cv2.imshow('License Plate Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()