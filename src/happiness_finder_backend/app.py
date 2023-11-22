import io
import os

from flask import Flask, request
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

model = YOLO("model/best.pt")  # pretrained YOLOv8x model
model(os.path.join(app.root_path, "bus.jpg"))


@app.route("/api/detect", methods=["POST"])
def detect():
    image = request.data
    image = Image.open(io.BytesIO(image))

    results = model.predict(image, imgsz=1280, half=True)
    results = results[0]
    results = results.tojson()

    return results


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
