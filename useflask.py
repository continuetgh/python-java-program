from flask import Flask,make_response, jsonify,request
import base64

# 配置全局app
app = Flask(__name__)
file_path="D:/Yolov5/testyolov5/test_image"
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file.save('path_to_save/' + file.filename)  # 保存文件到服务器本地，替换 'path_to_save/' 为您想要保存的路径
        return 'File uploaded successfully'

@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.json
    if 'image_data' not in data:
        return 'No image data received'
    image_data = data['image_data']
    # 将 base64 编码的图片数据解码并保存到服务器本地
    image = base64.b64decode(image_data)
    with open('path_to_save/image.png', 'wb') as f:  # 替换 'path_to_save/' 和 'image.png' 为您想要保存的路径和文件名
        f.write(image)
    return 'Image saved successfully'

if __name__ == '__main__':
    app.run(debug=True)