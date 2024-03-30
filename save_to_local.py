from flask import Flask,jsonify,request
import os

app = Flask(__name__)
@app.route('/upload',methods=['GET','POST'])
def uploads():
    img= request.files.get('face')#接受图片
    name = request.form.get('name')
    name = str(name) +'.jpg'
    img.save(os.path.join('test_image',name))
    return 'success'
