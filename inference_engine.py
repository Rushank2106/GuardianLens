"""
Guardian Lens - Inference & Risk Scoring Engine
Author: Principal Systems Architect
Description:
  - Loads trained YOLO model checkpoint on Apple Silicon MPS device.
  - Performs object detection & risk scoring on street-view imagery.
  - Implements an explainable severity heuristic based on bounding box area ratio,
    class-specific baseline risk weight, and confidence score.
  - Outputs structured JSON response for integration with full-stack civic dashboards.
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union
from ultralytics import YOLO

ROOT_DIR = Path(__file__).parent.resolve()
DEFAULT_WEIGHTS = ROOT_DIR / "runs" / "detect" / "GuardianLens" / "weights" / "best.pt"

# Class Baseline Severity Weights (1.0 to 10.0 scale)
# Priority ordering: Structural road hazards > Traffic safety hazards > Public sanitation / aesthetics
CLASS_SEVERITY_WEIGHTS: Dict[str, float] = {
    "pothole": 9.5,
    "damaged_road": 8.5,
    "broken_road_sign": 8.0,
    "illegal_parking": 7.0,
    "mixed_issues": 6.5,
    "vandalism": 5.0,
    "littering_garbage": 4.0
}


class GuardianLensInference:
    def __init__(self, model_path: Union[str, Path] = DEFAULT_WEIGHTS, device: str = None):
        """Initializes the inference engine and loads weights onto target device."""
        self.model_path = Path(model_path)
        assert self.model_path.exists(), f"Error: Model checkpoint not found at '{self.model_path}'!"

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"⚡ [Guardian Lens Inference] Loading model '{self.model_path.name}' on device '{self.device}'...")
        self.model = YOLO(str(self.model_path))

    def predict(self, image_input: Union[str, Path, np.ndarray], conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Runs inference on input image and computes explainable severity risk metrics.
        Returns structured JSON dictionary containing:
          - image_path / source identifier
          - detections list (class_name, confidence, bbox, area_ratio, risk_score)
          - overall_risk_score (1-100)
          - primary_hazard (highest severity issue detected)
          - severity_level (Low, Medium, High, Critical)
        """
        results = self.model.predict(
            source=image_input,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )[0]

        img_h, img_w = results.orig_shape
        total_img_area = float(img_w * img_h)

        detections_list: List[Dict[str, Any]] = []
        raw_risk_scores: List[float] = []

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = results.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0].item())

            # Bounding box coords in xyxy format
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [round(v, 2) for v in xyxy]
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            box_area = box_w * box_h

            area_ratio = box_area / total_img_area if total_img_area > 0 else 0.0

            # Explainable Risk Scoring Formula:
            # Risk = Confidence * BaseClassWeight * (1.0 + 1.5 * sqrt(AreaRatio))
            base_weight = CLASS_SEVERITY_WEIGHTS.get(cls_name.lower(), 5.0)
            area_multiplier = 1.0 + 1.5 * np.sqrt(area_ratio)
            box_risk = conf * base_weight * area_multiplier * 10.0

            raw_risk_scores.append(box_risk)

            detections_list.append({
                "class_id": cls_id,
                "label": cls_name,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
                "area_ratio": round(area_ratio, 4),
                "item_risk_score": round(box_risk, 2)
            })

        # Calculate Overall Risk Score (1-100)
        if not detections_list:
            overall_risk_score = 0
            primary_hazard = "None"
            severity_level = "Clear"
        else:
            # Sort detections by item_risk_score descending
            detections_list.sort(key=lambda d: d["item_risk_score"], reverse=True)
            primary_hazard = detections_list[0]["label"]

            # Aggregate composite score: max single hazard score + 30% of remaining hazards
            max_score = detections_list[0]["item_risk_score"]
            secondary_sum = sum(d["item_risk_score"] for d in detections_list[1:])
            composite = max_score + 0.3 * secondary_sum
            overall_risk_score = min(100, int(round(composite)))

            if overall_risk_score >= 75:
                severity_level = "Critical"
            elif overall_risk_score >= 50:
                severity_level = "High"
            elif overall_risk_score >= 25:
                severity_level = "Medium"
            else:
                severity_level = "Low"

        output_data = {
            "image": str(image_input) if isinstance(image_input, (str, Path)) else "ndarray_input",
            "image_dimensions": {"width": img_w, "height": img_h},
            "detection_count": len(detections_list),
            "primary_hazard": primary_hazard,
            "overall_risk_score": overall_risk_score,
            "severity_level": severity_level,
            "detections": detections_list
        }

        return output_data


if __name__ == "__main__":
    # Test script entrypoint
    engine = GuardianLensInference()
    print("Inference Engine module loaded successfully.")
