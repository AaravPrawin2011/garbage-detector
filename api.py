from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import io
import requests

app = Flask(__name__)
CORS(app)

# ================== CONFIGURATION ==================

# Model URL and local save path
MODEL_URL = "https://huggingface.co/Aarav2011/resnet18-garbage/resolve/main/resnet18_garbage.pth"
MODEL_PATH = "resnet18_garbage.pth"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

# Original class labels used during training
original_class_labels = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
                         'metal', 'paper', 'plastic', 'shoes', 'trash']

# Mapping for final API labels
mapping = {
    "glass": "metal",
    "cardboard": "cardboard/paper",
    "paper": "cardboard/paper"
}

def map_label(pred_label):
    return mapping.get(pred_label, pred_label)

api_class_labels = [map_label(label) for label in original_class_labels]

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

# ================== MODEL SETUP ==================

model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(original_class_labels))
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ================== TRANSFORMATIONS ==================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================== WEIGHT ESTIMATION ==================

def estimate_weight(image: Image.Image, label: str) -> float:
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    pixel_area = cv2.countNonZero(mask)
    real_area_cm2 = pixel_area * PIXEL_TO_CM2
    volume_cm3 = real_area_cm2 * THICKNESS
    mapped_label = map_label(label)
    density = CLASS_DENSITIES.get(mapped_label.lower(), 0.5)
    weight_g = volume_cm3 * density
    return round(weight_g, 2)

# ================== FLASK ROUTES ==================

@app.route("/", methods=["GET", "POST"])
def index():
    html_form = """
    <!doctype html>
    <html>
    <head><title>Garbage Detection API</title></head>
    <body style="text-align: center; font-family: sans-serif;">
        <h1>🟢 Garbage Detection API</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <br><br>
            <input type="submit" value="Upload and Detect">
        </form>
        {% if result %}
            <h2>Detected Category: <span style="color: green;">{{ result.prediction }}</span></h2>
            <h3>Confidence: {{ result.confidence }}</h3>
            <h3>Estimated Weight: {{ result.estimated_weight }}</h3>
        {% endif %}
    </body>
    </html>
    """
    result = None
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 400

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]
            predicted_idx = probs.argmax().item()
            predicted_label = original_class_labels[predicted_idx]
            final_label = map_label(predicted_label)
            confidence = probs[predicted_idx].item()

        weight_estimate = estimate_weight(image, predicted_label)
        weight_str = f"{round(weight_estimate/1000, 2)} kg" if weight_estimate >= 1000 else f"{weight_estimate} g"

        result = {
            "prediction": final_label,
            "confidence": f"{confidence * 100:.2f}%",
            "estimated_weight": weight_str
        }

        return render_template_string(html_form, result=result)

    return render_template_string(html_form)

# ================== RUN APP ==================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
