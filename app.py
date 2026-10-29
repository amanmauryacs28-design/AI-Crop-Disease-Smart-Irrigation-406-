
from flask import Flask, render_template, request
from pathlib import Path
from PIL import Image
import tensorflow as tf
import json
import numpy as np
import os
import uuid

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_disease_model.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

model = None
class_names = []

if MODEL_PATH.exists() and CLASS_PATH.exists():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, "r") as file:
        class_names = json.load(file)
    print("AI model loaded successfully!")
else:
    print("Model files not found. Train the model first.")


def predict_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)
    index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][index]) * 100

    return class_names[index], confidence


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":
        if model is None:
            error = "AI model files are missing. Please check the models folder."
        else:
            uploaded_file = request.files.get("leaf_image")

            if not uploaded_file or uploaded_file.filename == "":
                error = "Please select a leaf image."
            else:
                try:
                    image = Image.open(uploaded_file.stream)
                    image.verify()

                    uploaded_file.stream.seek(0)
                    filename = f"{uuid.uuid4().hex}.jpg"
                    save_path = UPLOAD_FOLDER / filename

                    image = Image.open(uploaded_file.stream).convert("RGB")
                    image.save(save_path, "JPEG")

                    result, confidence = predict_disease(save_path)
                    image_url = f"/static/uploads/{filename}"

                except Exception:
                    error = "Could not process this image. Please upload a valid image."

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_url=image_url,
        error=error
    )

@app.route("/irrigation", methods=["GET", "POST"])
def irrigation():
    moisture = None
    recommendation = None
    pump_status = None
    error = None

    if request.method == "POST":
        try:
            moisture = float(request.form.get("moisture", ""))

            if not 0 <= moisture <= 100:
                error = "Please enter a value between 0 and 100."
                moisture = None
            elif moisture < 30:
                recommendation = "Soil is dry. Irrigation is recommended."
                pump_status = "ON (simulated)"
            elif moisture < 60:
                recommendation = "Soil moisture is moderate. Monitor the soil."
                pump_status = "OFF (simulated)"
            else:
                recommendation = "Soil has sufficient moisture. Irrigation is not needed."
                pump_status = "OFF (simulated)"

        except (ValueError, TypeError):
            error = "Please enter a valid moisture percentage."

    return render_template(
        "irrigation.html",
        moisture=moisture,
        recommendation=recommendation,
        pump_status=pump_status,
        error=error
    )



if __name__ == "__main__":
    app.run(debug=True)
