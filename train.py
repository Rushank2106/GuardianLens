"""
Guardian Lens - Automated Training Pipeline
Author: Principal Systems Architect
Description:
  - Configures Ultralytics YOLOv8 / YOLOv11 object detection model for local training.
  - Target hardware: Apple Silicon GPU via PyTorch Metal Performance Shaders (device='mps').
  - Implements early stopping (patience=3) monitoring val/mAP50-95.
  - Tailored augmentations for class imbalance: mosaic=1.0, mixup=0.15, fliplr=0.5, flipud=0.0.
  - Automatically verifies best checkpoint saving at 'runs/detect/GuardianLens/weights/best.pt'.
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path(__file__).parent.resolve()
YAML_PATH = ROOT_DIR / "guardian_lens.yaml"


def verify_hardware():
    """Validates PyTorch MPS (Metal Performance Shaders) availability on Apple Silicon."""
    print("=" * 70)
    print("🔍 [Guardian Lens] Hardware & Environment Diagnostics")
    print("=" * 70)

    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Apple Silicon GPU detected and PyTorch MPS acceleration is AVAILABLE.")
    elif torch.cuda.is_available():
        device = 0
        print("✅ NVIDIA CUDA GPU detected.")
    else:
        device = "cpu"
        print("⚠️ Warning: MPS/CUDA GPU not available. Falling back to CPU.")

    return device


def run_training():
    """Executes automated model training pipeline."""
    assert YAML_PATH.exists(), f"Error: YAML configuration '{YAML_PATH}' not found. Run prepare_data.py first!"

    device = verify_hardware()

    print(f"\n🚀 Launching Ultralytics YOLO Training on Device: '{device}'")
    print("=" * 70)

    # Initialize model with lightweight pre-trained YOLO weights
    model = YOLO("yolov8n.pt")

    # Training parameters as specified
    results = model.train(
        data=str(YAML_PATH),
        epochs=30,             # Comprehensive training run
        batch=16,              # Efficient batch size for Apple Silicon unified memory
        imgsz=640,             # Standardized resolution
        device=device,         # Apple Silicon MPS
        patience=3,            # Early stopping patience
        save=True,
        project="runs/detect",
        name="GuardianLens",
        exist_ok=True,
        # Augmentations for class imbalance & spatial invariants
        mosaic=1.0,
        mixup=0.15,
        fliplr=0.5,
        flipud=0.0,
        plots=True,
        verbose=True
    )

    best_weights_path = ROOT_DIR / "runs" / "detect" / "GuardianLens" / "weights" / "best.pt"
    assert best_weights_path.exists(), f"Error: Best model checkpoint was not found at '{best_weights_path}'!"

    print("\n" + "=" * 70)
    print("🎉 Training Completed Successfully!")
    print(f"🏆 Best Checkpoint Saved At: '{best_weights_path.resolve()}'")
    print("=" * 70)

    return best_weights_path


if __name__ == "__main__":
    run_training()
