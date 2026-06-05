
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Load FULL model (architecture + weights)
model = load_model("emotion_cnn_model.keras")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = img.reshape(1, 48, 48, 1)
    return img


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            processed_image = preprocess_image(image_path)
            result = model.predict(processed_image)

            emotion_index = np.argmax(result)
            prediction = emotion_labels[emotion_index]

    return render_template("index.html", prediction=prediction, image_path=image_path)


if __name__ == "__main__":
    app.run(debug=True)