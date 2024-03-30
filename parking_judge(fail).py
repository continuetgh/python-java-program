import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes

# 加载模型
model = torch.hub.load('.', 'custom', path='runs/train/exp13/weights/best.pt', source='local')

# 读取文件夹中的所有图片
image_folder = Path("D:/Yolov5/testyolov5/data/mydata/images")
image_files = list(image_folder.glob("*.jpg"))  # 假设图片格式为jpg

# 打开保存结果的文件
result_file = open("results01.txt", "w")

# 循环遍历每张图片
for image_file in tqdm(image_files):
    # 读取图片
    img = Image.open(image_file).convert('RGB')

    # 检测目标
    results = model(img)

    # 遍历每个检测到的目标
    for i, detections in enumerate(results.pred):
        if detections is not None:
            # 进行非最大抑制
            keep = non_max_suppression(detections, conf_thres=0.5, iou_thres=0.5)

            # 遍历非最大抑制后的结果
            for det in keep:
                # 对每个检测到的目标进行处理
                if det.dim() == 0:  # 检查是否为零维张量
                    det = det.unsqueeze(0)  # 如果是，则添加一个维度
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in det:
                    # 将坐标进行缩放
                    box = [int(coord) for coord in scale_boxes(img.size, [x1, y1, x2, y2], img.size).round()]

                    # 获取目标类别的名称
                    class_name = model.names[int(cls_pred)]

                    # 判断是否为车辆或停车位
                    if class_name in ['bike','car','van']:
                        # TODO: 在这里进行车辆判断逻辑，判断车辆是否正确地停放在了车位内部
                        # 提示：可以通过比较车辆和停车位的坐标信息来判断是否正确停放
                        is_parked_correctly = True  # 这里需要根据实际情况进行逻辑编写
                        if is_parked_correctly:
                            result_file.write(f"{image_file.name}: Car parked correctly\n")
                        else:
                            result_file.write(f"{image_file.name}: Car not parked correctly\n")

                    elif class_name == 'parking':
                        # 可以对停车位进行一些操作，如果需要的话
                        pass

# 关闭结果文件
result_file.close()
