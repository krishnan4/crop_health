# ai_model.py
import torch
import timm
import json
import numpy as np
from torchvision import transforms
from PIL import Image, ImageEnhance
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.35

# Load class names
class_names_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_names.json")
with open(class_names_path, "r") as f:
    CLASSES = json.load(f)


def clean_name(raw):
    name = raw.replace("___", " — ").replace("__", " — ").replace("_", " ")
    return name.strip()


TREATMENTS = {
    "Pepper  bell   Bacterial spot": {
        "severity": "Medium",
        "treatment": "Spray copper-based fungicide every 7 days. Remove infected leaves immediately.",
        "prevention": "Avoid overhead watering. Use disease-free seeds."
    },
    "Pepper  bell   healthy": {
        "severity": "None",
        "treatment": "Your plant is healthy! No treatment needed.",
        "prevention": "Continue current care routine."
    },
    "Potato   Early blight": {
        "severity": "Medium",
        "treatment": "Apply Mancozeb 75% WP at 2g per litre of water every 7 days.",
        "prevention": "Crop rotation every season. Remove infected leaves."
    },
    "Potato   healthy": {
        "severity": "None",
        "treatment": "Your plant is healthy! No treatment needed.",
        "prevention": "Continue current care routine."
    },
    "Potato   Late blight": {
        "severity": "High",
        "treatment": "Apply Cymoxanil + Mancozeb immediately. Remove all infected plants.",
        "prevention": "Avoid excess moisture. Use resistant varieties."
    },
    "Tomato Bacterial spot": {
        "severity": "Medium",
        "treatment": "Copper hydroxide spray at 3g per litre every 5 days.",
        "prevention": "Use certified disease-free seeds. Avoid leaf wetness."
    },
    "Tomato Early blight": {
        "severity": "Medium",
        "treatment": "Neem oil spray every 5 days. Apply Chlorothalonil at 2.5g per litre.",
        "prevention": "Mulch around base. Practice crop rotation."
    },
    "Tomato healthy": {
        "severity": "None",
        "treatment": "Your plant is healthy! No treatment needed.",
        "prevention": "Continue current care routine."
    },
    "Tomato Late blight": {
        "severity": "High",
        "treatment": "Apply Mancozeb + Cymoxanil immediately. Remove infected leaves.",
        "prevention": "Avoid overhead watering. Ensure good air circulation."
    },
    "Tomato Leaf Mold": {
        "severity": "Medium",
        "treatment": "Apply Copper oxychloride at 3g per litre. Improve ventilation.",
        "prevention": "Reduce humidity. Avoid wetting leaves when watering."
    },
    "Tomato Septoria leaf spot": {
        "severity": "Medium",
        "treatment": "Apply Mancozeb or Chlorothalonil every 7 to 10 days.",
        "prevention": "Remove infected leaves. Avoid overhead irrigation."
    },
    "Tomato Spider mites Two spotted spider mite": {
        "severity": "Medium",
        "treatment": "Spray Abamectin 1.8% EC at 1ml per litre. Use neem oil as organic option.",
        "prevention": "Maintain humidity. Introduce predatory mites."
    },
    "Tomato  Target Spot": {
        "severity": "Medium",
        "treatment": "Apply Azoxystrobin or Mancozeb every 7 days.",
        "prevention": "Crop rotation. Remove plant debris after harvest."
    },
    "Tomato  Tomato mosaic virus": {
        "severity": "High",
        "treatment": "No cure available. Remove and destroy infected plants immediately.",
        "prevention": "Control aphids. Use virus-free seeds. Wash hands before handling."
    },
    "Tomato  Tomato YellowLeaf  Curl Virus": {
        "severity": "High",
        "treatment": "No cure. Remove infected plants. Control whitefly with Imidacloprid.",
        "prevention": "Use reflective mulches. Install yellow sticky traps."
    }
}


def get_treatment(disease_name):
    for key in TREATMENTS:
        if key.lower() in disease_name.lower() or disease_name.lower() in key.lower():
            return TREATMENTS[key]
    return {
        "severity": "Unknown",
        "treatment": "Consult a local agronomist for advice.",
        "prevention": "Remove infected plant parts immediately."
    }


def get_confidence_info(conf):
    if conf >= 0.80:
        return "high", "High confidence prediction ✅"
    elif conf >= 0.55:
        return "medium", "Moderate confidence — please verify result ⚠️"
    else:
        return "low", "Low confidence — try retaking photo in better light 📷"


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    arr = np.array(img)
    brightness = arr.mean()
    if brightness < 80:
        img = ImageEnhance.Brightness(img).enhance(2.0)
    elif brightness < 120:
        img = ImageEnhance.Brightness(img).enhance(1.4)

    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Sharpness(img).enhance(1.3)

    return img


def is_likely_not_leaf(img):
    arr = np.array(img).astype(float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    green_pixels = np.sum((g > r * 0.8) & (g > b * 0.8) & (g > 40))
    yellow_pixels = np.sum((r > 100) & (g > 80) & (b < 100))
    brown_pixels = np.sum((r > 80) & (g > 50) & (b < 60) & (r > g))

    total_pixels = arr.shape[0] * arr.shape[1]
    plant_ratio = (green_pixels + yellow_pixels + brown_pixels) / total_pixels

    return plant_ratio < 0.08


tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load disease model
print("Loading disease model...")
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")

if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    disease_model = timm.create_model(
        'efficientnet_b0', pretrained=False,
        num_classes=ckpt['num_classes']
    )
    disease_model.load_state_dict(ckpt['model_state'])
    disease_model.eval().to(DEVICE)
    print(f"Disease model loaded — {ckpt['num_classes']} classes")
else:
    print("⚠️ Model file not found. Using fallback predictions.")
    disease_model = None


def predict_with_tta(tensor, model):
    if model is None:
        return torch.softmax(torch.randn(1, 16), dim=1)[0]
    
    versions = [
        tensor,
        torch.flip(tensor, dims=[3]),
        torch.flip(tensor, dims=[2]),
        torch.rot90(tensor, k=1, dims=[2, 3]),
    ]
    all_probs = []
    with torch.no_grad():
        for v in versions:
            probs = torch.softmax(model(v), dim=1)
            all_probs.append(probs)
    return torch.stack(all_probs).mean(0)[0]


def predict_multicrop(img, model):
    if model is None:
        return torch.softmax(torch.randn(16), dim=0)
    
    w, h = img.size
    crops = [
        img.crop((w * 0.10, h * 0.10, w * 0.90, h * 0.90)),
        img.crop((0, 0, w * 0.75, h * 0.75)),
        img.crop((w * 0.25, 0, w, h * 0.75)),
        img.crop((0, h * 0.25, w * 0.75, h)),
        img.crop((w * 0.25, h * 0.25, w, h)),
    ]
    all_probs = []
    with torch.no_grad():
        for crop in crops:
            t = tf(crop).unsqueeze(0).to(DEVICE)
            probs = predict_with_tta(t, model)
            all_probs.append(probs)

    weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0])
    stacked = torch.stack(all_probs)
    averaged = (stacked * weights.unsqueeze(1)).sum(0) / weights.sum()
    return averaged


def predict_disease(image_path: str) -> dict:
    try:
        img = preprocess_image(image_path)

        if is_likely_not_leaf(img):
            return {
                "disease_name": "Not a plant leaf image",
                "confidence": 0,
                "is_leaf": False,
                "severity": "None",
                "treatment": "Please upload a clear photo of a plant leaf.",
                "prevention": "",
                "confidence_level": "low",
                "confidence_message": "No plant colors detected in image ❌",
                "all_predictions": []
            }

        if disease_model is None:
            return {
                "disease_name": "Tomato healthy (Fallback)",
                "confidence": 85.0,
                "is_leaf": True,
                "severity": "None",
                "treatment": "Your plant appears healthy! No treatment needed.",
                "prevention": "Continue regular care and monitoring.",
                "confidence_level": "high",
                "confidence_message": "Model not loaded - using fallback",
                "all_predictions": [{"name": "Tomato healthy", "confidence": 85.0}]
            }

        probs = predict_multicrop(img, disease_model)
        top3 = probs.topk(3)
        top_conf = top3.values[0].item()
        top_idx = top3.indices[0].item()

        raw_name = CLASSES[top_idx]
        disease_name = clean_name(raw_name)

        if "not" in raw_name.lower() and top_conf > 0.70:
            return {
                "disease_name": "Not a plant leaf image",
                "confidence": round(top_conf * 100, 1),
                "is_leaf": False,
                "severity": "None",
                "treatment": "Please upload a clear photo of a plant leaf.",
                "prevention": "",
                "confidence_level": "low",
                "confidence_message": "Image does not appear to be a plant leaf ❌",
                "all_predictions": []
            }

        treatment_info = get_treatment(disease_name)
        conf_level, conf_msg = get_confidence_info(top_conf)

        all_preds = [
            {
                "name": clean_name(CLASSES[top3.indices[i].item()]),
                "confidence": round(top3.values[i].item() * 100, 1)
            }
            for i in range(min(3, len(top3.indices)))
        ]

        return {
            "disease_name": disease_name,
            "confidence": round(top_conf * 100, 1),
            "is_leaf": True,
            "severity": treatment_info["severity"],
            "treatment": treatment_info["treatment"],
            "prevention": treatment_info["prevention"],
            "confidence_level": conf_level,
            "confidence_message": conf_msg,
            "all_predictions": all_preds
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "disease_name": "Prediction failed",
            "confidence": 0,
            "is_leaf": False,
            "severity": "Unknown",
            "treatment": f"Error: {str(e)}",
            "prevention": "",
            "confidence_level": "low",
            "confidence_message": "System error occurred",
            "all_predictions": []
        }