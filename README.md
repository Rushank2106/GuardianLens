# Guardian Lens 🛡️🔍
> **Automated Civic Issue & Infrastructure Defect Detection System**  
> Built with YOLOv8, PyTorch, and Apple Silicon MPS / CUDA acceleration for real-time civic problem identification (potholes, garbage littering, vandalism, damaged roads, and broken signage).

---

## 🌟 Overview
**Guardian Lens** is an AI-powered computer vision system designed to analyze public infrastructure and cleanliness issues. It categorizes civic defects into 7 key categories to assist municipal authorities and citizens in automated reporting and triage:

1. `vandalism`
2. `littering_garbage`
3. `pothole`
4. `broken_road_sign`
5. `damaged_road`
6. `mixed_issues`
7. `illegal_parking`

---

## 🏗️ Architecture & Features
- **YOLOv8 Object Detection**: High-performance real-time target bounding box detection.
- **Hardware Acceleration**: Built with native PyTorch **MPS (Metal Performance Shaders)** support for Apple Silicon M-Series GPUs, alongside NVIDIA CUDA & CPU fallback.
- **Automated Dataset Pipeline**: `prepare_data.py` normalizes raw annotations, handles bounding box scaling, creates train/val/test splits, and updates `guardian_lens.yaml`.
- **Custom Training Pipeline**: `train.py` features spatial augmentations (mosaic, mixup, fliplr) tailored to address civic dataset class imbalances.
- **Inference Engine**: `inference_engine.py` runs batch inference on single images, videos, or full directories, exporting annotated visual outputs and JSON defect metadata summaries.

---

## 🚀 Quick Start

### 1. Installation & Environment Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Rushank2106/GuardianLens.git
cd GuardianLens

# Create and activate virtual environment (optional)
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

### 2. Dataset Preparation
Organize your raw images and annotations under `data/`, then run the automated dataset structuring script:
```bash
python prepare_data.py
```
This builds standard YOLO-formatted annotations in `dataset/` and verifies `guardian_lens.yaml`.

---

### 3. Model Training
Launch automated YOLOv8 model training with Apple Silicon GPU acceleration:
```bash
python train.py
```
*Trained model checkpoints and metrics will be saved in `runs/detect/GuardianLens/`.*

---

### 4. Running Inference
Run defect detection on an input image or video stream:
```bash
python inference_engine.py --source path/to/image_or_video.jpg
```

---

## 📂 Project Structure
```
GuardianLens/
├── guardian_lens.yaml              # Dataset configuration & class label mappings
├── prepare_data.py                 # Dataset processing & train/val split script
├── train.py                        # Training pipeline with MPS/CUDA acceleration
├── inference_engine.py             # Inference runner & report generator
├── requirements.txt                # Python dependencies
├── Sewa First Innovation Challenge1.pdf # Project documentation & challenge brief
└── README.md                       # Project documentation
```

---

## ⚡ Requirements
- **Python**: 3.9+
- **PyTorch**: `>=2.0.0` (Apple Silicon MPS / CUDA supported)
- **Ultralytics**: `>=8.0.0`
- **OpenCV**: `>=4.8.0`
- **Numpy, Pillow, PyYAML, Scikit-Learn**

---

## 📜 License
This project is developed for civic infrastructure monitoring and community innovation challenges.
