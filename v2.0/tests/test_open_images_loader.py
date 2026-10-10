"""
Open Images fine-tuning loader: MIDs resolved from the official
class-descriptions CSV (both formats), YOLO labels written in the COCO id
space ObjectDetector filters on (one shared mapping).

Run: pytest v2.0/tests/test_open_images_loader.py
"""

import csv
import sys
from pathlib import Path

import pytest

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))

from perception.study_objects import (  # noqa: E402
    COCO80_NAMES,
    COCO_ID_TO_STUDY,
    OPEN_IMAGES_REFERENCE_MIDS,
)
from training.datasets.open_images_loader import (  # noqa: E402
    OpenImagesObjectDataset,
    bbox_row_to_yolo,
    convert_bbox_annotations,
    load_class_descriptions,
    resolve_study_mids,
)

# Excerpt of the official file (verified rows), incl. the MIDs the old code
# hard-coded, which are other classes.
_ROWS = [
    ("/m/011k07", "Tortoise"), ("/m/01knjb", "Billboard"), ("/m/04c0y", "Kangaroo"),
    ("/m/01mzpv", "Chair"), ("/m/0bt_c3", "Book"), ("/m/050k8", "Mobile phone"),
    ("/m/01c648", "Laptop"), ("/m/020lf", "Computer mouse"), ("/m/01m2v", "Computer keyboard"),
    ("/m/07cx4", "Telephone"),
]


def _desc_csv(path: Path, header: bool) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["LabelName", "DisplayName"])
        w.writerows(_ROWS)
    return path


@pytest.mark.parametrize("header", [False, True])
def test_descriptions_both_formats(tmp_path, header):
    d = load_class_descriptions(_desc_csv(tmp_path / "d.csv", header))
    assert d["/m/0bt_c3"] == "Book" and "LabelName" not in d


def test_mids_resolved_by_name_into_coco_ids(tmp_path):
    d = load_class_descriptions(_desc_csv(tmp_path / "d.csv", True))
    m = resolve_study_mids(d)
    assert m == {"/m/0bt_c3": 73, "/m/050k8": 67, "/m/01c648": 63, "/m/020lf": 64, "/m/01m2v": 66}
    assert set(m) == set(OPEN_IMAGES_REFERENCE_MIDS)
    for wrong in ("/m/01knjb", "/m/04c0y", "/m/01mzpv"):       # Billboard / Kangaroo / Chair
        assert wrong not in m


def test_missing_class_raises():
    with pytest.raises(ValueError, match="Book"):
        resolve_study_mids({"/m/050k8": "Mobile phone"})


def test_convert_and_filter(tmp_path):
    d = load_class_descriptions(_desc_csv(tmp_path / "d.csv", False))
    bbox = tmp_path / "train-annotations-bbox.csv"
    with open(bbox, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ImageID", "Source", "LabelName", "Confidence",
                                          "XMin", "XMax", "YMin", "YMax"])
        w.writeheader()
        w.writerow(dict(ImageID="img1", Source="x", LabelName="/m/0bt_c3", Confidence=1,
                        XMin=0.1, XMax=0.3, YMin=0.2, YMax=0.6))
        w.writerow(dict(ImageID="img1", Source="x", LabelName="/m/01knjb", Confidence=1,   # billboard
                        XMin=0.1, XMax=0.3, YMin=0.2, YMax=0.6))
        w.writerow(dict(ImageID="img2", Source="x", LabelName="/m/050k8", Confidence=1,
                        XMin=0.5, XMax=0.7, YMin=0.5, YMax=0.9))
    counts = convert_bbox_annotations(bbox, d, tmp_path / "labels" / "train")
    assert counts == {73: 1, 67: 1}
    line = (tmp_path / "labels" / "train" / "img1.txt").read_text().split()
    assert line[0] == "73" and [float(v) for v in line[1:]] == pytest.approx([0.2, 0.4, 0.2, 0.4])
    (tmp_path / "images" / "train").mkdir(parents=True)
    for n in ("img1", "img2", "img3"):
        (tmp_path / "images" / "train" / f"{n}.jpg").write_bytes(b"")
    (tmp_path / "labels" / "train" / "img3.txt").write_text("0 0.5 0.5 0.1 0.1\n")   # person only
    ds = OpenImagesObjectDataset(str(tmp_path), "train")
    assert [p[0].stem for p in ds.iter_yolo_pairs()] == ["img1", "img2"]
    yaml = ds.write_dataset_yaml(tmp_path / "data.yaml").read_text()
    assert "nc: 80" in yaml and "  73: book" in yaml and "  67: cell phone" in yaml


def test_degenerate_box_skipped():
    assert bbox_row_to_yolo({"ImageID": "a", "LabelName": "/m/0bt_c3", "XMin": 0.5, "XMax": 0.5,
                             "YMin": 0.1, "YMax": 0.2}, {"/m/0bt_c3": 73}) is None


def test_shared_mapping_matches_coco_names():
    assert len(COCO80_NAMES) == 80
    assert {COCO80_NAMES[k] for k in COCO_ID_TO_STUDY} == {"laptop", "mouse", "keyboard", "cell phone", "book"}


def test_object_detector_uses_shared_mapping():
    pytest.importorskip("ultralytics")
    from perception import object_detector
    assert object_detector.STUDY_CLASSES is COCO_ID_TO_STUDY


def test_mapping_matches_the_shipped_detector_weights():
    pytest.importorskip("ultralytics")
    path = V2_ROOT / "models" / "yolov8s-objects.pt"
    if not path.exists():
        pytest.skip("models/yolov8s-objects.pt not downloaded")
    from ultralytics import YOLO
    names = YOLO(str(path)).names
    assert tuple(names[i] for i in range(80)) == COCO80_NAMES
