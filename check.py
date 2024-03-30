from flask import Flask, request, jsonify
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

# 定义模型加载和图片检测函数
def load_model_and_detect(image_path):
    # 加载模型并进行图片检测，这里使用伪代码代替
    # 假设模型返回一个检测结果的字典
    detection_result = {'objects': ['bike','car','van','parking']}
    return detection_result

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    # 保存上传的文件到指定路径
    upload_folder = 'test_image'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # 调用模型进行图片检测
    detection_result = load_model_and_detect(file_path)

    # 返回检测结果给微信小程序
    return jsonify(detection_result)

if __name__ == '__main__':
    app.run(debug=True)
