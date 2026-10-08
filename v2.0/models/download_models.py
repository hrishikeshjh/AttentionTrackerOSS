"""
download_models.py — Download YOLOv8 weights for v2.0 server.

Run from the repo root or the v2.0/ directory:
    python3 v2.0/models/download_models.py
    # or
    cd v2.0 && python3 models/download_models.py

Requirements: ultralytics  (pip install ultralytics)
"""

import shutil
import sys
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(__file__).parent


def _find_cached(model_name: str) -> Optional[Path]:
    """
    Search every location where ultralytics might have cached the weights.
    The download destination has changed across ultralytics versions:
      < 8.1  → cwd / model_name
      8.1+   → ~/.config/Ultralytics/  (Linux/macOS)
      8.1+   → %APPDATA%/Ultralytics/  (Windows)
      some   → ~/.ultralytics/assets/
    """
    candidates = [
        Path(model_name),                                           # cwd
        Path.home() / ".config" / "Ultralytics" / model_name,     # 8.1+ macOS/Linux
        Path.home() / "AppData" / "Roaming" / "Ultralytics" / model_name,  # Windows
        Path.home() / ".ultralytics" / "assets" / model_name,     # older
        Path.home() / ".cache"  / "ultralytics" / model_name,     # some builds
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def download(model_name: str, dest_name: str) -> None:
    dest = MODELS_DIR / dest_name
    if dest.exists():
        print(f"[download_models] Already present: {dest}")
        return
    print(f"[download_models] Downloading {model_name} → {dest_name} ...")
    try:
        from ultralytics import YOLO  # type: ignore
        YOLO(model_name)   # triggers download into ultralytics cache

        src = _find_cached(model_name)
        if src is None:
            print(f"  WARNING: could not locate downloaded weights for {model_name}")
            print(f"  Search paths checked — copy {model_name} to {dest} manually.")
            return

        shutil.copy(src, dest)
        print(f"  Saved to {dest}")
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)


def export_onnx(pt_path: Path, imgsz: int = 640) -> None:
    onnx_path = pt_path.with_suffix(".onnx")
    if onnx_path.exists():
        print(f"[download_models] ONNX already present: {onnx_path.name}")
        return
    print(f"[download_models] Exporting {pt_path.name} → ONNX …")
    try:
        from ultralytics import YOLO  # type: ignore
        m = YOLO(str(pt_path))
        m.export(format="onnx", imgsz=imgsz, simplify=True)
        # ultralytics writes <name>.onnx next to the .pt file
        src = pt_path.with_suffix(".onnx")
        if src.exists() and src != onnx_path:
            src.rename(onnx_path)
        print(f"  Saved to {onnx_path}")
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)


if __name__ == "__main__":
    # Person detection (v2.0 uses medium for accuracy)
    download("yolov8m.pt",       "yolov8m.pt")
    # Study-context object detection (small for speed)
    download("yolov8s.pt",       "yolov8s-objects.pt")
    # Pose estimation
    download("yolov8m-pose.pt",  "yolov8m-pose.pt")

    # Optionally export to ONNX for onnxruntime inference
    for pt_name in ["yolov8m.pt", "yolov8s-objects.pt", "yolov8m-pose.pt"]:
        pt_path = MODELS_DIR / pt_name
        if pt_path.exists():
            export_onnx(pt_path)

    print("\n[download_models] All models ready.")
