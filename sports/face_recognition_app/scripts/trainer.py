import os
import cv2
import numpy as np
import json
import joblib
import pywt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from collections import Counter

def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray) / 255.0
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # remove approximation
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = np.uint8(imArray_H * 255)
    return imArray_H

# Initialize data
X, y = [], []
class_dict = {}
dataset_path = "dataset/cropped"
class_id = 0

# Optional face detection setup (uncomment if needed)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("📂 Preparing training data...")

for class_dir in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_dir)
    if not os.path.isdir(class_path):
        continue

    print(f"➡️ Processing class: {class_dir}")
    class_dict[class_dir] = class_id

    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to load image: {img_path}")
            continue

        try:
            # Optional face detection (uncomment if needed)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # if len(faces) == 0:
            #     continue
            # (x, y_, w, h) = faces[0]
            # img = img[y_:y_+h, x:x+w]

            if img.shape[2] != 3:
                print(f"⚠️ Skipping non-RGB image: {img_path}")
                continue

            scaled_raw_img = cv2.resize(img, (32, 32))
            img_har = w2d(img, 'db1', 5)
            scaled_img_har = cv2.resize(img_har, (32, 32))

            combined_img = np.vstack((
                scaled_raw_img.reshape(32 * 32 * 3, 1),
                scaled_img_har.reshape(32 * 32, 1)
            ))
            X.append(combined_img.flatten())
            y.append(class_id)
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            continue

    class_id += 1

if not X:
    raise Exception("❌ No training data found. Check if your dataset is in 'dataset/cropped/' and contains valid images.")

# Save class dictionary
with open("class_dictionary.json", "w") as f:
    json.dump(class_dict, f)

# Convert and shuffle
X = np.array(X)
y = np.array(y)
X, y = shuffle(X, y, random_state=42)

# Show class distribution
print("📊 Class distribution:", Counter(y))

# Train and save model
print("🧠 Training model...")
model = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='linear', probability=True))
])
model.fit(X, y)

joblib.dump(model, "saved_model.pkl", compress=3)

print(f"✅ Training complete. Trained on {len(X)} samples across {len(class_dict)} classes.")
print("📁 Model saved to 'saved_model.pkl'")
print("📁 Class dictionary saved to 'class_dictionary.json'")
