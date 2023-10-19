import base64
import io

from flask import Flask, request
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

model = YOLO("model/best.pt")  # pretrained YOLOv8x model


@app.route("/api/detect", methods=["POST"])
def detect():
    json_data = request.get_json()
    image = json_data["image"].split(",")[1]
    image = base64.b64decode(image)
    image = Image.open(io.BytesIO(image))

    results = model(image)
    results = results[0]
    results = results.tojson()

    return results


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
