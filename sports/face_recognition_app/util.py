# util.py
import os
import numpy as np
import cv2
import pywt
import joblib
import json

# -------------------
# Global variables
# -------------------
__model = None
__class_name_to_number = {}
__class_number_to_name = {}

# -------------------
# Wavelet transformation
# -------------------
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray) / 255.0
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # zero out approximation coefficients
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = np.uint8(imArray_H * 255)
    return imArray_H

# -------------------
# Load model + class dict
# -------------------
def load_saved_artifacts():
    global __model, __class_name_to_number, __class_number_to_name

    # Calculate absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'ml_model')
    dict_path = os.path.join(model_dir, 'class_dictionary.json')
    model_path = os.path.join(model_dir, 'saved_model.pkl')

    # Load class dictionary
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Missing: {dict_path}")
    with open(dict_path, 'r') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    # Load the trained model
    if __model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing: {model_path}")
        __model = joblib.load(model_path)

# -------------------
# Classify an image from path
# -------------------
def classify_image(image_path):
    if not os.path.exists(image_path):
        return "Invalid image path"

    img = cv2.imread(image_path)
    if img is None:
        return "Failed to load image."

    # Face detection using Haar cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "No face detected"

    # Crop the first detected face
    x, y, w, h = faces[0]
    roi = img[y:y+h, x:x+w]

    # Feature extraction
    try:
        raw_resized = cv2.resize(roi, (32, 32))
        wav_img = w2d(roi, mode='db1', level=5)
        wav_resized = cv2.resize(wav_img, (32, 32))

        combined = np.vstack((
            raw_resized.reshape(32*32*3, 1),
            wav_resized.reshape(32*32, 1)
        )).reshape(1, 4096).astype(float)

        prediction_num = __model.predict(combined)[0]
        return __class_number_to_name.get(prediction_num, "Unknown")
    except Exception as e:
        return f"Error during classification: {e}"

# -------------------
# Initialization for Django apps
# -------------------
# Load artifacts immediately when imported
try:
    load_saved_artifacts()
except Exception as e:
    # If running manage.py checks, just log silently or raise if critical
    print(f"[util.py] Warning: model artifacts not loaded -> {e}")
