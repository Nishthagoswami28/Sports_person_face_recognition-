# 🏆 Sports Legends Identifier

**Sports Legends Identifier** is a web application built using Django and OpenCV that uses a Machine Learning model to recognize famous sports personalities from uploaded face images.

---

## 🚀 Features

- 📸 Upload an image and automatically detect the face
- 🧠 Predict the identity using a trained ML model
- 🖼️ Display prediction with face preview
- 📂 Support for multiple known sports figures
- 🧾 Sample known faces displayed on homepage
- 💾 Upload and temporary storage using Django media handling

---

## 🛠️ Technologies Used

- ⚙️ **Django** – High-level Python web framework
- 🤖 **OpenCV** – Face detection and image processing
- 🧮 **NumPy** – Numerical operations
- 📦 **Joblib** – Model serialization
- 🌊 **PyWavelets** – Wavelet transform for feature extraction
- 🗂️ **JSON** – Class-label mapping for predictions

---

## 📥 Installation

### 📌 Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/sports-legends-identifier.git
cd sports-legends-identifier
````

### 📌 Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # For Linux/macOS
venv\Scripts\activate         # For Windows
```

### 📌 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### 📌 Step 4: Run the Server

```bash
python manage.py runserver

```

Now open your browser and go to 👉 `http://127.0.0.1:8000/`

---

## 🧠 Supported Faces

The model currently recognizes the following sports legends:

* 🏏 MS Dhoni
* ⚽ Lionel Messi
* 🏏 Virat Kohli
* 🎾 Roger Federer
* ⚽ Cristiano Ronaldo

---

## 📸 Screenshots

### 🏠 Home Page

![Home Page](\media\uploads\home.png)

---

## 📃 License

This project is licensed under the **MIT License** – feel free to use, modify, and distribute.

---

## 💬 About

This project was built to explore the intersection of machine learning, computer vision, and web development. It demonstrates real-time face detection and classification using pre-trained models and Python web frameworks.

---

## 🔗 Resources

* Django Docs: [https://docs.djangoproject.com/](https://docs.djangoproject.com/)
* OpenCV Docs: [https://docs.opencv.org/](https://docs.opencv.org/)
* PyWavelets: [https://pywavelets.readthedocs.io/](https://pywavelets.readthedocs.io/)

```
