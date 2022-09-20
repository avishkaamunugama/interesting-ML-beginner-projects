from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from ds_comp.prediction import *
import string

app = Flask(__name__)


# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send_text/', methods=['POST', 'GET'])
def upload_image():
    text = request.form['text']
    processed_text = text.lower()
    processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
    print(processed_text)
    obj = Prediction(processed_text)
    prediction = obj.make_pred()

    print(prediction)

    return render_template('index.html', prediction=prediction, text=text)


if __name__ == "__main__":
    app.run(debug=True)
