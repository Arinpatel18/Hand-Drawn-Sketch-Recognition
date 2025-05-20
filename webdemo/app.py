from flask import Flask, render_template, request
from ann_predict import predict_from_image
from svm_predict import predict_svm
from logistic_regression_predict import predict_logistic_regression
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    model_type = request.form['model']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    if model_type == 'ann':
        pred_class = predict_from_image(filepath)
    elif model_type == 'svm':
        pred_class = predict_svm(filepath)
    elif model_type == 'logistic':
        pred_class = predict_logistic_regression(filepath)
    else:
        pred_class = "Model not implemented yet"

    return render_template('index.html', prediction=pred_class)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
