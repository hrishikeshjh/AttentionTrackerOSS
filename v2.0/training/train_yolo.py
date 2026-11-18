"""
train_yolo.py — Fine-tune YOLOv8 open weights for classroom and study object detection.
Reference: parameters.md §2 (YOLOv8 Open-Weights Vision Backbone Tuning)

Usage:
    python3 v2.0/training/train_yolo.py \\
        --model yolov8s.pt \\
        --data classroom_objects.yaml \\
        --epochs 120 \\
        --batch 16 \\
        --imgsz 640 \\
        --optimizer AdamW \\
        --lr0 0.0008 \\
        --output-dir runs/cognen_objects

Supports:
    - Person detection (yolov8m.pt)
    - Study objects (yolov8s.pt / yolov8n.pt)
    - Keypoint pose estimation (yolov8m-pose.pt)
    - Dynamic backbone layer freezing for small datasets
    - Automatic ONNX export with simplification
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cognen.train_yolo")

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 open weights for CognEn")
    # Model and Data
    p.add_argument("--model", type=str, default="yolov8s.pt",
                   help="Base model checkpoint or open-weights filename (e.g. yolov8s.pt, yolov8m.pt)")
    p.add_argument("--data", type=str, required=True,
                   help="Path to dataset YAML file (e.g. classroom_objects.yaml)")
    p.add_argument("--output-dir", type=str, default=str(V2_ROOT / "runs" / "train_yolo"),
                   help="Project directory to store training runs and weights")
    p.add_argument("--name", type=str, default="yolov8_finetune",
                   help="Run name within output-dir")

    # Basic Training Hyperparameters
    p.add_argument("--epochs", type=int, default=120, help="Total training epochs")
    p.add_argument("--patience", type=int, default=25, help="Early stopping patience (epochs)")
    p.add_argument("--batch", type=int, default=16, help="Batch size (reduce to 8 or 4 if VRAM-constrained)")
    p.add_argument("--imgsz", type=int, default=640, help="Image resolution (640 for desk, 1280 for CCTV)")
    p.add_argument("--device", type=str, default="", help="CUDA device index or 'cpu' (default: auto)")

    # Optimizer & Learning Rate Schedule
    p.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "SGD", "Adam"],
                   help="Optimizer (AdamW recommended for domain transfer)")
    p.add_argument("--lr0", type=float, default=0.0008, help="Initial learning rate")
    p.add_argument("--lrf", type=float, default=0.01, help="Final learning rate fraction (lr_min = lr0 * lrf)")
    p.add_argument("--momentum", type=float, default=0.90, help="Optimizer momentum / beta1")
    p.add_argument("--weight-decay", type=float, default=0.001, help="L2 regularization weight decay")
    p.add_argument("--warmup-epochs", type=float, default=5.0, help="Warmup epochs")

    # Loss Weights
    p.add_argument("--box", type=float, default=8.5, help="Bounding box regression loss gain")
    p.add_argument("--cls", type=float, default=1.2, help="Classification loss gain")
    p.add_argument("--dfl", type=float, default=1.5, help="Distribution Focal Loss gain")
    p.add_argument("--pose", type=float, default=14.0, help="Pose keypoint loss gain (pose models only)")
    p.add_argument("--kobj", type=float, default=1.5, help="Keypoint objectness loss gain")

    # Domain Augmentation Hyperparameters (parameters.md §2.3)
    p.add_argument("--mosaic", type=float, default=0.5, help="Mosaic augmentation probability")
    p.add_argument("--close-mosaic", type=int, default=15, help="Disable mosaic during final N epochs")
    p.add_argument("--mixup", type=float, default=0.1, help="Mixup augmentation probability")
    p.add_argument("--copy-paste", type=float, default=0.2, help="Copy-paste augmentation probability")
    p.add_argument("--erasing", type=float, default=0.3, help="Random erasing probability (simulates occlusion)")
    p.add_argument("--degrees", type=float, default=10.0, help="Rotation degrees range")
    p.add_argument("--hsv-h", type=float, default=0.015, help="HSV hue jitter")
    p.add_argument("--hsv-s", type=float, default=0.4, help="HSV saturation jitter")
    p.add_argument("--hsv-v", type=float, default=0.3, help="HSV value/brightness jitter")
    p.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    p.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability (must be 0.0)")

    # Transfer Learning & Export
    p.add_argument("--freeze", type=int, default=10,
                   help="Number of backbone layers to freeze (10 for neck+head, 15 for head only, 0 for none)")
    p.add_argument("--export-onnx", action="store_true", default=True,
                   help="Export final best weights to ONNX format upon completion")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate arguments and environment without starting training")
    return p


def train_yolo(args: argparse.Namespace) -> dict[str, Any]:
    """Execute YOLOv8 fine-tuning with open weights."""
    logger.info("Initializing YOLO fine-tuning with base model: %s", args.model)
    logger.info("Dataset config: %s | Epochs: %d | Batch: %d | Imgsz: %d",
                args.data, args.epochs, args.batch, args.imgsz)

    if args.dry_run:
        logger.info("[Dry Run] Configuration verified successfully.")
        return {"status": "dry_run", "model": args.model, "data": args.data}

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error("ultralytics is not installed. Install with: pip install ultralytics")
        raise

    model = YOLO(args.model)

    train_kwargs: dict[str, Any] = {
        "data": args.data,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "erasing": args.erasing,
        "degrees": args.degrees,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "project": args.output_dir,
        "name": args.name,
        "save": True,
    }

    if args.device:
        train_kwargs["device"] = args.device

    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze

    # If pose estimation model, attach pose loss parameters
    if "pose" in args.model.lower():
        train_kwargs["pose"] = args.pose
        train_kwargs["kobj"] = args.kobj

    results = model.train(**train_kwargs)
    logger.info("Training completed. Results: %s", results)

    best_pt = Path(args.output_dir) / args.name / "weights" / "best.pt"
    onnx_path = None

    if args.export_onnx and best_pt.exists():
        logger.info("Exporting best checkpoint (%s) to ONNX format...", best_pt)
        best_model = YOLO(str(best_pt))
        exported_file = best_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        onnx_path = str(exported_file)
        logger.info("ONNX export ready at: %s", onnx_path)

    return {
        "best_checkpoint": str(best_pt) if best_pt.exists() else None,
        "onnx_export": onnx_path,
        "epochs_trained": args.epochs,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_yolo(args)


if __name__ == "__main__":
    main()
