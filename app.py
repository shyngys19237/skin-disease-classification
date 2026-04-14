import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import gradio as gr

MODEL_PATH = "best_model.pth"
CLASS_NAMES_PATH = "class_names.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

num_classes = len(class_names)

model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def predict(image):
    if image is None:
        return {"No image uploaded": 1.0}

    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top3_idx = np.argsort(probs)[::-1][:3]

    results = {}
    for idx in top3_idx:
        results[class_names[idx]] = float(probs[idx])

    return results

title = "Skin Condition Classification Demo"
description = (
    "Upload a skin image and the model will return the top-3 predicted classes.\n\n"
    "Not a medical diagnosis. For educational/demo purposes only. "
    "Consult a dermatologist for professional evaluation."
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title=title,
    description=description,
)

if __name__ == "__main__":
    demo.launch()
