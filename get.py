import torch
import sys
import os
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

app = Flask(__name__)
# 从本地加载自定义的YOLOv5模型
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
model_path = str(ROOT)  # yolov5根目录，需要转换为字符串类型
weight_path = str(ROOT / "runs"/"train"/"exp13"/"weights"/"best.pt""")  # 设置正确的权重文件路径
model = torch.hub.load(model_path, "custom", weight_path, source="local", force_reload=True)


# results.show()  # 这两句用于看一下模型检测结果

@app.route('/upload', methods=['POST'])
def recognize_image():
    # 检查是否存在图像文件
    if 'file' not in request.files:
        return '未选择文件'

    file = request.files['file']
    if file.filename == '':
        return '未选择文件'
    # 保存文件到本地
    save_path = model_path + '/test_image'
    file.save(save_path + file.filename)
    imgs = [str(save_path) + file.filename]  # 设置正确的图像文件路径
    img_path = str(save_path) + file.filename
    results = model(imgs)
    results.print()
    results.save(save_path + 'detections')
    res = results.pandas().xyxy
    confidence = 0
    tagname = ''
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    for i, boxes in enumerate(res):
        print(f"第 {i + 1} 个结果：")
        for _, row in boxes.iterrows():
            if row['confidence'] > 0.5:
                # 计算模型准确率
                confidence = '{:.2%}'.format(row['confidence'])
                tagname = row['name']
                print(row['confidence'])
                # 打印预测结果
                print(f"预测标签: {row['name']}")

    # 计算模型准确率
    # accuracy = calculate_accuracy([(img, pred_index)])  # 使用辅助函数计算准确率
    # print("模型准确率: {:.2%}".format(accuracy))
    # 处理检测结果并返回
    # TODO: 返回处理后的检测结果
    if confidence == 0:
        msg = '当前没有脉动'
    else:
        msg = '识别完成物体为:' + tagname + '的可信度:' + confidence
    return jsonify({'code': 200, 'msg': msg})


def get_image_path(image_folder, image_file):
    # 获取当前脚本文件的路径
    script_dir = Path(__file__).resolve().parent

    # 构建图片文件的完整路径
    image_path = script_dir / image_folder / image_file

    return image_path


@app.route('/test', methods=['GET'])
def test_detect_objects():
    # 读取本地图片文件
    imgs = [str(model_path) + '/test_image/t_1.jpg']  # 设置正确的图像文件路径
    results = model(imgs)
    results.print()
    res = results.pandas().xyxy
    confidence = 0;
    for i, boxes in enumerate(res):
        print(f"第 {i + 1} 个结果：")
        for _, row in boxes.iterrows():
            if row['confidence'] > 0.5:
                confidence = '{:.2%}'.format(row['confidence'])
                print(row['confidence'])
                # 打印预测结果
                print(f"预测标签: {row['name']}")

    # 计算模型准确率
    # accuracy = calculate_accuracy([(img, pred_index)])  # 使用辅助函数计算准确率
    # print("模型准确率: {:.2%}".format(accuracy))

    # 处理检测结果并返回
    # TODO: 返回处理后的检测结果

    return '识别完成物体为:的可信度:' + confidence


if __name__ == '__main__':
    app.run(host='127.0.0.1')