"""
Guardian Lens - Dataset Preparation & Alignment Pipeline
Author: Principal Computer Vision Architect
Description:
  - Validates and sanitizes raw images from 7 civic issue categories.
  - Standardizes resolution to 640x640 via letterbox padding (preserves aspect ratio without distortion).
  - Generates YOLO-compliant label bounding boxes.
  - Performs deterministic stratified split (70% Train, 20% Val, 10% Test).
  - Emits guardian_lens.yaml configuration for Ultralytics YOLO.
"""

import os
import shutil
import random
import cv2
import numpy as np
import yaml
from pathlib import Path

# --- Configuration & Seed setup ---
SEED = 42
TARGET_SIZE = 640
PAD_COLOR = (114, 114, 114)  # Standard YOLO neutral gray

ROOT_DIR = Path(__file__).parent.resolve()
DATA_SOURCE_DIR = ROOT_DIR / "data"
DATASET_OUTPUT_DIR = ROOT_DIR / "dataset"
YAML_OUTPUT_PATH = ROOT_DIR / "guardian_lens.yaml"

# Class Mapping: Relative Source Subdirectory -> (Normalized Class Name, Class ID)
CATEGORY_MAPPING = {
    os.path.join("Public Cleanliness + Environmental Issues", "Vandalism Issues"): ("vandalism", 0),
    os.path.join("Public Cleanliness + Environmental Issues", "Littering Garbage on Public Places Issues"): ("littering_garbage", 1),
    os.path.join("Road Issues", "Pothole Issues"): ("pothole", 2),
    os.path.join("Road Issues", "Broken Road Sign Issues"): ("broken_road_sign", 3),
    os.path.join("Road Issues", "Damaged Road issues"): ("damaged_road", 4),
    os.path.join("Road Issues", "Mixed Issues"): ("mixed_issues", 5),
    os.path.join("Road Issues", "Illegal Parking Issues"): ("illegal_parking", 6),
}

NAMES = {
    0: "vandalism",
    1: "littering_garbage",
    2: "pothole",
    3: "broken_road_sign",
    4: "damaged_road",
    5: "mixed_issues",
    6: "illegal_parking"
}


def letterbox_image(image: np.ndarray, target_shape=(TARGET_SIZE, TARGET_SIZE), color=PAD_COLOR):
    """
    Resizes and letterbox-pads an image to target_shape while preserving aspect ratio.
    Returns:
      padded_image: uint8 numpy array of size target_shape
      bbox: tuple of normalized YOLO coords (x_center, y_center, width, height)
    """
    h_orig, w_orig = image.shape[:2]
    w_target, h_target = target_shape

    scale = min(w_target / w_orig, h_target / h_orig)
    w_new = int(round(w_orig * scale))
    h_new = int(round(h_orig * scale))

    # Perform high quality resize
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (w_new, h_new), interpolation=interp)

    # Calculate padding offsets
    pad_x = (w_target - w_new) // 2
    pad_y = (h_target - h_new) // 2

    # Create padded background
    padded = np.full((h_target, w_target, 3), color, dtype=np.uint8)
    padded[pad_y:pad_y + h_new, pad_x:pad_x + w_new] = resized

    # Normalized bounding box coordinates relative to 640x640
    x_center = (pad_x + w_new / 2.0) / w_target
    y_center = (pad_y + h_new / 2.0) / h_target
    norm_w = w_new / w_target
    norm_h = h_new / h_target

    return padded, (x_center, y_center, norm_w, norm_h)


def prepare_dataset():
    """Executes full dataset sanitation, letterboxing, stratified splitting, and label generation."""
    print("=" * 70)
    print("🚀 [Guardian Lens] Starting Dataset Sanitation & Alignment Pipeline")
    print("=" * 70)

    assert DATA_SOURCE_DIR.exists(), f"Error: Source directory '{DATA_SOURCE_DIR}' does not exist!"

    # Clean / Reset output dataset directory
    if DATASET_OUTPUT_DIR.exists():
        print(f"Cleaning existing dataset directory at '{DATASET_OUTPUT_DIR}'...")
        shutil.rmtree(DATASET_OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        (DATASET_OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    total_processed = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_summary = {cls_name: {"total": 0, "train": 0, "val": 0, "test": 0} for cls_name, _ in CATEGORY_MAPPING.values()}

    for rel_subpath, (class_name, class_id) in CATEGORY_MAPPING.items():
        src_category_dir = DATA_SOURCE_DIR / rel_subpath
        assert src_category_dir.exists(), f"Error: Category subpath '{src_category_dir}' not found!"

        image_files = [
            f for f in os.listdir(src_category_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        image_files.sort()  # Ensure deterministic order before shuffling
        random.shuffle(image_files)

        n_total = len(image_files)
        n_train = int(round(n_total * 0.70))
        n_val = int(round(n_total * 0.20))
        # n_test gets remainder to guarantee 100% assignment
        n_test = n_total - n_train - n_val

        splits = (
            [("train", img) for img in image_files[:n_train]] +
            [("val", img) for img in image_files[n_train:n_train + n_val]] +
            [("test", img) for img in image_files[n_train + n_val:]]
        )

        class_summary[class_name]["total"] = n_total
        print(f"📂 Category '{class_name}' (ID: {class_id}): Total={n_total} -> Train={n_train}, Val={n_val}, Test={n_test}")

        for split_name, filename in splits:
            src_img_path = src_category_dir / filename
            img = cv2.imread(str(src_img_path))

            if img is None:
                print(f"⚠️ Warning: Skipping unreadable/corrupt image: {src_img_path}")
                continue

            # Standardize image via letterboxing to 640x640
            padded_img, (xc, yc, w, h) = letterbox_image(img)

            # Generate unique clean file stem
            file_stem = f"{class_name}_{total_processed:05d}"

            dst_img_path = DATASET_OUTPUT_DIR / "images" / split_name / f"{file_stem}.jpg"
            dst_lbl_path = DATASET_OUTPUT_DIR / "labels" / split_name / f"{file_stem}.txt"

            # Save processed image
            cv2.imwrite(str(dst_img_path), padded_img)

            # Write YOLO format label file: <class_id> <xc> <yc> <w> <h>
            label_content = f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
            with open(dst_lbl_path, "w") as f_lbl:
                f_lbl.write(label_content)

            total_processed += 1
            split_counts[split_name] += 1
            class_summary[class_name][split_name] += 1

    print("\n" + "=" * 70)
    print(f"✅ Sanitation & Partition Complete!")
    print(f"Total Processed Images: {total_processed}")
    print(f"Split distribution: Train={split_counts['train']} ({split_counts['train']/total_processed*100:.1f}%), "
          f"Val={split_counts['val']} ({split_counts['val']/total_processed*100:.1f}%), "
          f"Test={split_counts['test']} ({split_counts['test']/total_processed*100:.1f}%)")
    print("=" * 70)

    # Generate guardian_lens.yaml config
    yaml_config = {
        "path": str(DATASET_OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": NAMES
    }

    with open(YAML_OUTPUT_PATH, "w") as f_yaml:
        yaml.dump(yaml_config, f_yaml, sort_keys=False)

    print(f"📝 Generated YOLO config dataset file at: '{YAML_OUTPUT_PATH.resolve()}'")


if __name__ == "__main__":
    prepare_dataset()
