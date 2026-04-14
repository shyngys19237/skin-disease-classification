# Skin Condition Classification Demo

A simple AI-powered skin image classification demo built with **ResNet-50** and deployed on **Hugging Face Spaces**.

## Live Demo
[Open the demo](https://huggingface.co/spaces/shyngys879/skin-condition-demo)

## Project Overview
This project is a prototype for **skin condition classification from images**.  
The model takes a skin image as input and returns the **top-3 predicted classes**.

It is intended as an **educational / research demo** and **not** as a medical diagnosis tool.

## Dataset
The model was trained on a dermatology image dataset with **22 skin condition classes**:

- Acne
- Actinic_Keratosis
- Benign_tumors
- Bullous
- Candidiasis
- DrugEruption
- Eczema
- Infestations_Bites
- Lichen
- Lupus
- Moles
- Psoriasis
- Rosacea
- Seborrh_Keratoses
- SkinCancer
- Sun_Sunlight_Damage
- Tinea
- Unknown_Normal
- Vascular_Tumors
- Vasculitis
- Vitiligo
- Warts

## Model
- Backbone: **ResNet-50**
- Initialization: pretrained on ImageNet
- Framework: **PyTorch**
- Demo interface: **Gradio**

## Test Performance
Baseline performance on the held-out test set:
- **Top-1 accuracy:** 40.3%
- **Macro F1:** 34.1%
- **Weighted F1:** 38.9%
- **Top-3 accuracy:** 67.7%

## Live Demo Features
- Upload a skin image
- Receive the **top-3 predicted classes**
- Simple browser-based interface
- Fast inference through Gradio

## Files
- `app.py` — Gradio application for inference
- `requirements.txt` — Python dependencies
- `class_names.json` — class label mapping
- `README.md` — project description and usage instructions
- `test_metrics.json` — saved evaluation metrics on the test set

## Limitations
- This is **not a medical diagnosis system**
- The model is a **baseline prototype**
- Performance is limited and may not generalize well to real-world clinical images
- The dataset is multi-class and visually challenging, with class imbalance and overlapping conditions
- The current demo should be viewed as an educational ML project, not a healthcare product

## Ethical Notice
This tool is for **educational and demonstration purposes only**.  
It should **not** be used for self-diagnosis or clinical decision-making.  
Please consult a qualified dermatologist for professional medical evaluation.

## Tech Stack
- Python
- PyTorch
- Torchvision
- Gradio
- NumPy
- Pillow

## Future Improvements
- Full fine-tuning on GPU
- Better class balancing
- More robust evaluation
- Uncertainty estimation
- Improved clinically meaningful grouping of skin conditions
