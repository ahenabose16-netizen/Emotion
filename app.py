
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model("emotion_cnn_model.keras")

emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral'
]


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
        file = request.files.get("image")

        if file and file.filename:

            # Delete all image files in static folder
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                if (
                    os.path.isfile(file_path)
                    and filename.lower().endswith(
                        ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
                    )
                ):
                    os.remove(file_path)

            # Save new image
            image_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                file.filename
            )
            file.save(image_path)

            # Predict emotion
            processed_image = preprocess_image(image_path)
            result = model.predict(processed_image)

            emotion_index = np.argmax(result)
            prediction = emotion_labels[emotion_index]

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)