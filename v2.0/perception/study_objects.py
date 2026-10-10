"""
study_objects.py — the ONE class mapping for study-context objects.

ObjectDetector (perception/object_detector.py) runs a COCO-80 YOLO model and
keeps the COCO class ids below. A detector fine-tuned on Open Images
(training/datasets/open_images_loader.py) is trained with the SAME COCO ids
in its YOLO label files, so the fine-tuned weights drop into ObjectDetector
unchanged.

Open Images classes are resolved by DisplayName from the official
class-descriptions CSV at run time (open_images_loader.load_class_descriptions).
OPEN_IMAGES_REFERENCE_MIDS records the MIDs those names had in the official
files, verified 2026-10-10 against
https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
and .../v7/oidv7-class-descriptions-boxable.csv (both list the same five).
The previous hard-coded MIDs were wrong: /m/01knjb is "Billboard",
/m/04c0y "Kangaroo", /m/01mzpv "Chair".
"""

from __future__ import annotations

# COCO-80 class id -> CognEn object name (what ObjectDetection.cls carries)
COCO_ID_TO_STUDY: dict[int, str] = {
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "phone",      # COCO "cell phone"
    73: "book",
}
STUDY_TO_COCO_ID: dict[str, int] = {v: k for k, v in COCO_ID_TO_STUDY.items()}

# Open Images DisplayName -> CognEn object name
OPEN_IMAGES_NAME_TO_STUDY: dict[str, str] = {
    "Book":              "book",
    "Mobile phone":      "phone",
    "Laptop":            "laptop",
    "Computer mouse":    "mouse",
    "Computer keyboard": "keyboard",
}

# Reference only (verified, see module docstring); the loader resolves MIDs
# from the CSV it is given and cross-checks against these.
OPEN_IMAGES_REFERENCE_MIDS: dict[str, str] = {
    "/m/0bt_c3": "Book",
    "/m/050k8":  "Mobile phone",
    "/m/01c648": "Laptop",
    "/m/020lf":  "Computer mouse",
    "/m/01m2v":  "Computer keyboard",
}

# COCO-80 names in id order (YOLO dataset.yaml for a COCO-id label space).
COCO80_NAMES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)
