"""
open_images_loader.py — Open Images v7 loader for object detector fine-tuning.

Used to fine-tune YOLOv8-nano/s on the study-context object classes:
  book (LabelName /m/01knjb), cell phone (/m/050k8), laptop (/m/01c648),
  computer mouse (/m/04c0y), computer keyboard (/m/01mzpv)

Download: https://storage.googleapis.com/openimages/web/download_v7.html

Expected directory layout:
    <root>/
        images/
            train/ val/ test/
        labels/
            train/ val/ test/      ← YOLO-format .txt files (class cx cy w h)
        metadata/
            classes.csv            ← LabelName,DisplayName mapping

This loader yields (image_path, annotations) pairs suitable for YOLO training.
It does NOT produce behavior windows — use it to call Ultralytics directly.

Usage:
    from training.datasets.open_images_loader import OpenImagesObjectDataset
    ds = OpenImagesObjectDataset("/data/openimages", split="train")
    for img_path, labels_path in ds.iter_yolo_pairs():
        ...  # pass directly to ultralytics YOLO.train(data=yaml)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger("cognen.datasets.openimages")

# Open Images LabelName → YOLO class index for CognEn study objects
_OI_LABEL_TO_IDX: dict[str, int] = {
    "/m/01knjb":  0,  # book
    "/m/050k8":   1,  # cell phone
    "/m/01c648":  2,  # laptop
    "/m/04c0y":   3,  # computer mouse
    "/m/01mzpv":  4,  # computer keyboard
}

_OI_IDX_TO_NAME: dict[int, str] = {
    0: "book",
    1: "phone",
    2: "laptop",
    3: "mouse",
    4: "keyboard",
}


class OpenImagesObjectDataset:
    """
    Iterates over Open Images image / label pairs for YOLO fine-tuning.

    Generates a YOLO-compatible dataset.yaml on demand.
    """

    CLASSES = list(_OI_IDX_TO_NAME.values())

    def __init__(self, root: str, split: str = "train") -> None:
        self.root    = Path(root)
        self.split   = split
        self._img_dir   = self.root / "images"   / split
        self._label_dir = self.root / "labels"   / split

    # ── public ───────────────────────────────────────────────────────────────

    def iter_yolo_pairs(self) -> Generator[tuple[Path, Path], None, None]:
        """
        Yield (image_path, label_path) for every image that has a non-empty
        YOLO annotation file containing at least one study-context object.
        """
        if not self._img_dir.exists():
            logger.warning("Open Images image dir not found: %s", self._img_dir)
            return
        if not self._label_dir.exists():
            logger.warning("Open Images label dir not found: %s", self._label_dir)
            return

        for img_path in sorted(self._img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = self._label_dir / (img_path.stem + ".txt")
            if not label_path.exists() or label_path.stat().st_size == 0:
                continue
            yield img_path, label_path

    def write_dataset_yaml(self, output_path: str | Path) -> Path:
        """
        Write a YOLO dataset.yaml compatible with Ultralytics training.

        Args:
            output_path: Destination path for the YAML file.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        lines = [
            f"path: {self.root}",
            "train: images/train",
            "val:   images/val",
            "test:  images/test",
            "",
            f"nc: {len(self.CLASSES)}",
            f"names: {self.CLASSES}",
        ]
        output_path.write_text("\n".join(lines))
        logger.info("Wrote dataset YAML to %s", output_path)
        return output_path

    @staticmethod
    def filter_labels(label_path: Path) -> list[str]:
        """
        Read a YOLO .txt label file and return only rows belonging to
        study-context class indices.
        """
        valid_ids = set(_OI_IDX_TO_NAME.keys())
        kept = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id in valid_ids:
                kept.append(line)
        return kept
