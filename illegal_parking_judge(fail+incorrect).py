import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
# 指定本地模型文件路径
from utils.general import non_max_suppression, scale_boxes
import torchvision.transforms as transforms

# 加载模型
model = torch.hub.load('.', 'custom', path='runs/train/exp13/weights/best.pt', source='local')

# 读取文件夹中的所有图片
image_folder = Path("D:/Yolov5/testyolov5/data/mydata/images")
image_files = list(image_folder.glob("*.jpg"))  # 假设图片格式为jpg

# 打开保存结果的文件
result_file = open("results.txt", "w")

# 定义图像变换，将图像转换为张量
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # 调整图像大小为模型期望的大小
    transforms.ToTensor(),  # 转换为张量
])

# 获取模型的权重数据类型
model_weight_dtype = next(model.parameters()).dtype
# 循环遍历每张图片
for image_file in tqdm(image_files):
    # 读取图片
    img = Image.open(image_file).convert('RGB')

    # 对图像进行变换
    img = transform(img)

    # 添加一个批次维度
    img = img.unsqueeze(0)

    # 获取模型的权重数据类型
    model_weight_dtype = next(model.parameters()).dtype

    # 将输入张量的数据类型转换为与模型的权重数据类型相匹配的类型
    img = img.to(torch.float16 if model_weight_dtype == torch.float16 else torch.float32)

    # 检测目标
    results = model(img)

    # 进行非最大抑制
    results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5)
# 遍历每个检测到的目标
    # 遍历每个检测到的目标
    for detections in results:
        if detections is not None:
            # 对每个检测到的目标进行处理
            for detection in detections:
                if len(detection) == 7:  # 检查检测结果是否包含了期望的7个值
                    x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
                    # 将坐标进行缩放
                    box = [int(coord) for coord in scale_boxes(img.size, [x1, y1, x2, y2], img.size).round()]

                    # 获取目标类别的名称
                    class_name = model.names[int(cls_pred)]

                    # 判断是否为车辆或停车位
                    if class_name in ['car', 'bike', 'van']:
                        # TODO: 在这里进行车辆判断逻辑，判断车辆是否正确地停放在了车位内部
                        # 提示：可以通过比较车辆和停车位的坐标信息来判断是否正确停放
                        is_parked_correctly = True  # 这里需要根据实际情况进行逻辑编写
                        if is_parked_correctly:
                            result_file.write(f"{image_file.name}: vehicle parked correctly\n")
                        else:
                            result_file.write(f"{image_file.name}: vehicle not parked correctly\n")

                else:
                    print("Unexpected number of values in detection result.")



# 关闭结果文件
result_file.close()