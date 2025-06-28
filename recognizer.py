import numpy as np
import cv2
import pywt
import joblib
import json
import os

# -------------------------------
# Wavelet transform function
# -------------------------------
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray) / 255.0
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # Set approximation coefficients to zero
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = np.uint8(imArray_H * 255)
    return imArray_H

# -------------------------------
# Globals
# -------------------------------
__model = None
__class_name_to_number = {}
__class_number_to_name = {}

# -------------------------------
# Load saved model and class map
# -------------------------------
def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    with open("class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    if __model is None:
        __model = joblib.load("saved_model.pkl")

    print("Loading saved artifacts...done")

# -------------------------------
# Classify an image by path
# -------------------------------
def classify_image(image_path):
    if not os.path.exists(image_path):
        return "Invalid image path"

    img = cv2.imread(image_path)
    if img is None:
        return "Failed to load image."

    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected"

    (x, y, w, h) = faces[0]
    roi = img[y:y+h, x:x+w]

    try:
        scaled_raw_img = cv2.resize(roi, (32, 32))
        img_har = w2d(roi, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))

        combined_img = np.vstack((
            scaled_raw_img.reshape(32 * 32 * 3, 1),
            scaled_img_har.reshape(32 * 32, 1)
        ))

        final = combined_img.reshape(1, 4096).astype(float)
        prediction = __model.predict(final)[0]
        return __class_number_to_name[prediction]
    except Exception as e:
        return f"Prediction failed: {str(e)}"

# -------------------------------
# Entry point for testing
# -------------------------------
if __name__ == '__main__':
    load_saved_artifacts()
    result = classify_image("test_images/test4.jpg")  # Make sure this file exists
    print("Prediction:", result)
