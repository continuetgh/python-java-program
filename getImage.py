import torch
import sys
import os
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

app = Flask(__name__)

model = torch.hub.load('.', 'custom', path='runs/train/exp13/weights/best.pt', source='local',force_reload=True)

@app.route('/upload',methods=['POST'])
def recognize_image():
    if 'file' not in request.files:
        return '未选择文件'

    #读取上传的图像文件
    file = request.files['file']
    if file.filename == '':
        return '未选择文件'

    #保存图片到本地
    save_path= model_path+'/dataset/upload/'
    file.save(save_path+file.filename)
    imgs = [str(save_path)+file.filename]
    img_path = str(save_path) + file.filename
    results = model(img)
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
    #对预测结果进行预处理

    return jsonify({'code':200,'msg':msg})

@app.route('/test',methods=['GET'])
def test_detect_objects():
    imgs = [str(model_path)+'/dataset/t_1.jpg']
    results = model(imgs)


    return '识别完成物体为：'+tagname+'的可信度：'+confidence