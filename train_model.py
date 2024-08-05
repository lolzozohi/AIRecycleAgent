import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
import joblib

# Function to extract HOG features
def extract_hog_features(image):
    features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
    return features

# Load and preprocess data
def load_data(data_dir):
    labels = []
    features = []
    for label in ['recyclable', 'non_recyclable']:
        dir_path = os.path.join(data_dir, label)
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            image = cv2.imread(file_path)
            image = cv2.resize(image, (128, 128))
            hog_features = extract_hog_features(image)
            features.append(hog_features)
            labels.append(0 if label == 'non_recyclable' else 1)
    return np.array(features), np.array(labels)

# Main script
data_dir = 'path/to/your/data'
features, labels = load_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'logistic_model.pkl')
