"""
test_cvat_to_classroom_json.py — regression tests for the CVAT-to-
ClassroomDataset annotation converter (Part 1e dataset scaffolding).

Validated against a hand-built fixture CVAT XML export
(tests/fixtures/cvat_sample.xml), not real annotated footage -- no real
CVAT project/classroom recording exists in this environment. This proves
the converter logic and output format are correct so it's ready to use
once real annotation output exists.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.datasets.annotation_tools.cvat_to_classroom_json import (
    convert_cvat_xml,
    write_classroom_annotation,
)
from training.datasets.classroom_loader import ClassroomDataset

FIXTURE = Path(__file__).parent / "fixtures" / "cvat_sample.xml"


def test_convert_cvat_xml_produces_expected_persons_shape():
    ann = convert_cvat_xml(FIXTURE)
    assert len(ann["persons"]) == 2

    p0 = ann["persons"][0]
    assert p0["track_id"] == 0
    assert p0["seat_id"] == "seat_1"
    assert len(p0["frames"]) == 3
    assert p0["frames"][0]["behavior"] == "attentive_screen"
    assert p0["frames"][2]["behavior"] == "using_phone"
    assert p0["frames"][0]["bbox_xyxy"] == [10.0, 20.0, 110.0, 220.0]


def test_convert_cvat_xml_second_track():
    ann = convert_cvat_xml(FIXTURE)
    p1 = ann["persons"][1]
    assert p1["track_id"] == 1
    assert p1["seat_id"] == "seat_2"
    assert all(f["behavior"] == "attentive_reading" for f in p1["frames"])


def test_write_classroom_annotation_produces_valid_classroom_dataset_input():
    """The real end-to-end check: write the converted output, then load
    it with the ACTUAL ClassroomDataset loader (not a reimplementation of
    its expectations) to prove format compatibility."""
    with tempfile.TemporaryDirectory() as tmp:
        write_classroom_annotation(
            xml_path=FIXTURE, clip_id="clip_0001", split="train", fps=5.0, out_dir=tmp,
        )

        ann_path = Path(tmp) / "annotations" / "clip_0001.json"
        assert ann_path.exists()
        ann = json.loads(ann_path.read_text())
        assert ann["clip_id"] == "clip_0001"
        assert ann["fps"] == 5.0

        meta_path = Path(tmp) / "metadata.csv"
        assert meta_path.exists()
        with open(meta_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["clip_id"] == "clip_0001"
        assert rows[0]["split"] == "train"
        assert rows[0]["num_persons"] == "2"

        # Real integration check: ClassroomDataset must accept this
        # metadata.csv without erroring on _load_metadata().
        ds = ClassroomDataset(tmp, split="train")
        clip_ids = ds._load_metadata()
        assert clip_ids == ["clip_0001"]


def test_write_classroom_annotation_appends_multiple_clips():
    with tempfile.TemporaryDirectory() as tmp:
        write_classroom_annotation(FIXTURE, "clip_0001", "train", 5.0, tmp)
        write_classroom_annotation(FIXTURE, "clip_0002", "val", 5.0, tmp)

        meta_path = Path(tmp) / "metadata.csv"
        with open(meta_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert {r["clip_id"] for r in rows} == {"clip_0001", "clip_0002"}
