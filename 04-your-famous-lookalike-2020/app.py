from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from ds_comp.prediction import *
import os

PEOPLE_FOLDER = os.path.join('/static')
ALLOWED_EXTENTIONS = {'jpeg', 'jpg', 'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_image/', methods=['POST', 'GET'])
def upload_image():
    if request.method == 'POST':
        # checking for file
        if 'file' not in request.files:
            print('no file')
            return redirect('/')
        file = request.files['file']
        if file.filename == '':
            print('no selected file')
            return redirect('/')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            obj = Prediction(file)
            prediction = obj.make_pred()

            image = os.path.join(app.config['UPLOAD_FOLDER'], prediction[0])
            similarity = prediction[1]
            celebrity_name = prediction[2]
            return render_template('index.html', filename=filename, prediction=image, similarity=similarity, celebrity_name=celebrity_name)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
