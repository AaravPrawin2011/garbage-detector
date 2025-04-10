from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision.transforms as transforms
import cv2
import numpy as np
import requests
import io
import os

app = Flask(__name__)
CORS(app)

# ================== CONFIG ==================

MODEL_URL = "https://huggingface.co/Aarav2011/resnet18-garbage/resolve/main/resnet18_garbage.pth"
MODEL_PATH = "resnet18_garbage.pth"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)

class_labels = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
                'metal', 'paper', 'plastic', 'shoes', 'trash']

mapping = {
    "glass": "metal",
    "cardboard": "cardboard/paper",
    "paper": "cardboard/paper"
}

CLASS_DENSITIES = {
    'battery': 2.5,
    'biological': 1.0,
    'cardboard/paper': 0.69,
    'clothes': 0.4,
    'metal': 7.8,
    'plastic': 0.94,
    'shoes': 0.5,
    'trash': 0.5
}

PIXEL_TO_CM2 = 0.01
THICKNESS = 0.2

# ================== MODEL ==================

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_labels))
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def map_label(label):
    return mapping.get(label, label)

def estimate_weight(image: Image.Image, label: str) -> float:
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    pixel_area = cv2.countNonZero(mask)
    real_area_cm2 = pixel_area * PIXEL_TO_CM2
    volume_cm3 = real_area_cm2 * THICKNESS
    mapped_label = map_label(label)
    density = CLASS_DENSITIES.get(mapped_label.lower(), 0.5)
    return round(volume_cm3 * density, 2)

# ================== ROUTES ==================

@app.route("/", methods=["GET", "POST"])
def index():
    html = """
    <!doctype html>
    <html>
    <head><title>Garbage Detector</title></head>
    <body style="text-align:center; font-family:sans-serif;">
        <h1>Garbage Detection</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required><br><br>
            <input type="submit" value="Detect">
        </form>
        {% if result %}
            <h2>Prediction: <span style="color:green;">{{ result.prediction }}</span></h2>
            <h3>Confidence: {{ result.confidence }}</h3>
            <h3>Estimated Weight: {{ result.weight }}</h3>
        {% endif %}
    </body>
    </html>
    """
    result = None
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image"}), 400
        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]
            idx = probs.argmax().item()
            raw_label = class_labels[idx]
            label = map_label(raw_label)
            conf = probs[idx].item()
        weight = estimate_weight(image, raw_label)
        weight_str = f"{round(weight/1000, 2)} kg" if weight >= 1000 else f"{weight} g"
        result = {
            "prediction": label,
            "confidence": f"{conf * 100:.2f}%",
            "weight": weight_str
        }
        return render_template_string(html, result=result)
    return render_template_string(html)

if __name__ == "__main__":
    app.run()
