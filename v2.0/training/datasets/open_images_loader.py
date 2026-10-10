"""
open_images_loader.py — Open Images (v6/v7) loader for object detector fine-tuning.

Fine-tunes the study-context object detector (book, phone, laptop, mouse,
keyboard). Class ids written to the YOLO label files are the COCO-80 ids that
perception/object_detector.py filters on (single mapping:
perception/study_objects.py), so a fine-tuned checkpoint is a drop-in
replacement for models/yolov8s-objects.pt.

MIDs are NOT hard-coded: they are resolved by DisplayName from the official
class-descriptions CSV you pass in (class-descriptions-boxable.csv, v5/v6:
no header; oidv7-class-descriptions-boxable.csv: header
"LabelName,DisplayName"). The previous hard-coded MIDs pointed at
Billboard / Kangaroo / Chair.

Download: https://storage.googleapis.com/openimages/web/download_v7.html
  - oidv7-class-descriptions-boxable.csv
  - <split>-annotations-bbox.csv  (columns: ImageID, Source, LabelName,
    Confidence, XMin, XMax, YMin, YMax, IsOccluded, IsTruncated, ...;
    coordinates normalised to [0, 1])
  - the images

Directory layout produced / expected:
    <root>/
        images/<split>/<ImageID>.jpg
        labels/<split>/<ImageID>.txt      YOLO: "<coco_id> cx cy w h" (normalised)

Usage:
    from training.datasets.open_images_loader import (
        OpenImagesObjectDataset, convert_bbox_annotations, load_class_descriptions)
    desc = load_class_descriptions("oidv7-class-descriptions-boxable.csv")
    convert_bbox_annotations("train-annotations-bbox.csv", desc, "/data/oi/labels/train")
    ds = OpenImagesObjectDataset("/data/oi", split="train")
    ds.write_dataset_yaml("/data/oi/dataset.yaml")   # then ultralytics YOLO.train(data=...)
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Generator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # v2.0 root (perception/)

from perception.study_objects import (
    COCO80_NAMES,
    COCO_ID_TO_STUDY,
    OPEN_IMAGES_NAME_TO_STUDY,
    OPEN_IMAGES_REFERENCE_MIDS,
    STUDY_TO_COCO_ID,
)

logger = logging.getLogger("cognen.datasets.openimages")


def load_class_descriptions(csv_path: str | Path) -> dict[str, str]:
    """{MID: DisplayName} from an Open Images class-descriptions CSV (with or without header)."""
    out: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            mid, name = row[0].strip(), row[1].strip()
            if mid == "LabelName":          # v7 header
                continue
            out[mid] = name
    if not out:
        raise ValueError(f"no classes read from {csv_path}")
    return out


def resolve_study_mids(descriptions: dict[str, str]) -> dict[str, int]:
    """
    {MID: COCO class id} for the study objects, resolved by DisplayName.
    Raises if a study class is missing or ambiguous. Logs a warning when a
    MID differs from OPEN_IMAGES_REFERENCE_MIDS (a new Open Images release).
    """
    by_name: dict[str, list[str]] = defaultdict(list)
    for mid, name in descriptions.items():
        by_name[name].append(mid)
    out: dict[str, int] = {}
    for name, study in OPEN_IMAGES_NAME_TO_STUDY.items():
        mids = by_name.get(name, [])
        if len(mids) != 1:
            raise ValueError(f"Open Images class {name!r}: expected exactly one MID, found {mids}")
        mid = mids[0]
        if OPEN_IMAGES_REFERENCE_MIDS.get(mid) != name:
            logger.warning("Open Images %r has MID %s (reference table differs)", name, mid)
        out[mid] = STUDY_TO_COCO_ID[study]
    return out


def bbox_row_to_yolo(row: dict, mid_to_coco: dict[str, int]) -> tuple[str, str] | None:
    """(ImageID, YOLO line) for one annotations-bbox row, or None if not a study class."""
    cls = mid_to_coco.get((row.get("LabelName") or "").strip())
    if cls is None:
        return None
    x1, x2 = float(row["XMin"]), float(row["XMax"])
    y1, y2 = float(row["YMin"]), float(row["YMax"])
    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return row["ImageID"].strip(), f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}"


def convert_bbox_annotations(
    bbox_csv: str | Path,
    descriptions: dict[str, str],
    out_label_dir: str | Path,
) -> dict[int, int]:
    """
    Write one YOLO label file per image that has at least one study object.
    Returns {coco_id: n_boxes}.
    """
    mid_to_coco = resolve_study_mids(descriptions)
    per_image: dict[str, list[str]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    with open(bbox_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            conv = bbox_row_to_yolo(row, mid_to_coco)
            if conv is None:
                continue
            image_id, line = conv
            per_image[image_id].append(line)
            counts[int(line.split()[0])] += 1
    out = Path(out_label_dir)
    out.mkdir(parents=True, exist_ok=True)
    for image_id, lines in per_image.items():
        (out / f"{image_id}.txt").write_text("\n".join(lines) + "\n")
    logger.info("Wrote %d label files to %s: %s", len(per_image), out, dict(counts))
    return dict(counts)


class OpenImagesObjectDataset:
    """
    Iterates over Open Images image / label pairs for YOLO fine-tuning and
    writes a YOLO dataset.yaml in the COCO-80 id space.
    """

    CLASSES = [COCO_ID_TO_STUDY[k] for k in sorted(COCO_ID_TO_STUDY)]
    CLASS_IDS = sorted(COCO_ID_TO_STUDY)

    def __init__(self, root: str, split: str = "train") -> None:
        self.root    = Path(root)
        self.split   = split
        self._img_dir   = self.root / "images"   / split
        self._label_dir = self.root / "labels"   / split

    # ── public ───────────────────────────────────────────────────────────────

    def iter_yolo_pairs(self) -> Generator[tuple[Path, Path], None, None]:
        """
        Yield (image_path, label_path) for every image whose YOLO label file
        contains at least one study-context object.
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
            if not label_path.exists() or not self.filter_labels(label_path):
                continue
            yield img_path, label_path

    def write_dataset_yaml(self, output_path: str | Path) -> Path:
        """
        YOLO dataset.yaml in the COCO-80 label space (nc: 80, names by id) so
        the class ids match ObjectDetector.STUDY_CLASSES.
        """
        output_path = Path(output_path)
        lines = [
            f"path: {self.root}",
            "train: images/train",
            "val:   images/val",
            "test:  images/test",
            "",
            f"nc: {len(COCO80_NAMES)}",
            "names:",
        ]
        lines += [f"  {i}: {name}" for i, name in enumerate(COCO80_NAMES)]
        output_path.write_text("\n".join(lines) + "\n")
        logger.info("Wrote dataset YAML to %s", output_path)
        return output_path

    @staticmethod
    def filter_labels(label_path: Path) -> list[str]:
        """Rows of a YOLO .txt label file whose class id is a study-object COCO id."""
        valid_ids = set(COCO_ID_TO_STUDY)
        kept = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if int(parts[0]) in valid_ids:
                kept.append(line)
        return kept
