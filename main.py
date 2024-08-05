import os
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to extract HOG features
def extract_hog_features(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Load the trained model
model = joblib.load('logistic_model.pkl')

# Prediction function
def predict(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    hog_features = extract_hog_features(image)
    hog_features = np.array(hog_features).reshape(1, -1)
    prediction = model.predict(hog_features)
    return 'Recyclable' if prediction == 1 else 'Non-recyclable'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = predict(filepath, model)
            return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
