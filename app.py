"""
Melanoma Detection - Backend API
---------------------------------
Flask REST API that loads the trained CNN and serves predictions.

Endpoints:
    GET  /api/health          - health check
    POST /api/predict         - upload an image, get benign/malignant prediction
"""

import os
import sys
import io
import base64

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
from model import build_model

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model_weights.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
class_names = checkpoint["class_names"]

model = build_model(pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": class_names, "device": str(device)})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if "image" in request.files:
            file = request.files["image"]
            img = Image.open(file.stream).convert("RGB")
        elif request.is_json and "image_base64" in request.json:
            b64 = request.json["image_base64"]
            img_bytes = base64.b64decode(b64.split(",")[-1])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            return jsonify({"error": "No image provided. Send multipart 'image' or JSON 'image_base64'."}), 400

        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())

        result = {
            "prediction": class_names[pred_idx],
            "confidence": round(float(probs[pred_idx]) * 100, 2),
            "probabilities": {
                class_names[i]: round(float(probs[i]) * 100, 2) for i in range(len(class_names))
            },
            "disclaimer": "This is an educational demo model trained on synthetic sample "
                           "data, NOT a validated medical diagnostic tool."
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
