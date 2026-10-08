"""
cvat_to_classroom_json.py — convert a CVAT "CVAT for video 1.1" XML export
into the annotation format ClassroomDataset expects
(training/datasets/classroom_loader.py).

Nothing in this repo currently produces ClassroomDataset's expected input
format (<root>/annotations/<clip_id>.json + <root>/metadata.csv) -- this is
the bridge from a real annotation tool (CVAT, self-hosted, free) to that
format, so real classroom footage can actually be labeled and used once it
exists. Real footage/CVAT projects are NOT available in this environment;
this script is tooling only, validated against a hand-built fixture XML
(see tests/test_cvat_to_classroom_json.py), not real exported data.

Expected CVAT track structure per clip (one CVAT task per clip_id):
    <annotations>
      <meta><task><size>150</size></task></meta>
      <track id="0" label="student">
        <box frame="0" xtl="10.0" ytl="20.0" xbr="110.0" ybr="220.0">
          <attribute name="behavior">attentive_screen</attribute>
          <attribute name="seat_id">seat_1</attribute>
        </box>
        ...
      </track>
      ...
    </annotations>

Output (ClassroomDataset's expected schema):
    <out_dir>/annotations/<clip_id>.json:
        {
          "clip_id": "...", "fps": 5,
          "persons": [
            {"track_id": 0, "seat_id": "seat_1",
             "frames": [{"frame_idx": 0, "bbox_xyxy": [...], "behavior": "..."}]}
          ]
        }
    <out_dir>/metadata.csv row appended: clip_id,split,duration_sec,num_persons

Usage:
    python3 cvat_to_classroom_json.py \\
        --cvat-xml /path/to/clip_0001.xml \\
        --clip-id clip_0001 --split train --fps 5 --out-dir /data/classroom
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_BEHAVIOR = "distracted"  # matches classroom_loader.py's fallback


def _box_attributes(box_el: ET.Element) -> dict[str, str]:
    return {
        attr.get("name", ""): (attr.text or "").strip()
        for attr in box_el.findall("attribute")
    }


def convert_cvat_xml(xml_path: str | Path) -> dict:
    """
    Parse a CVAT-for-video-1.1 XML export into ClassroomDataset's
    per-clip annotation dict (everything except `clip_id`, which the
    caller supplies since CVAT track XML doesn't necessarily know its own
    clip name).
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    persons = []
    for track_el in root.findall("track"):
        track_id = int(track_el.get("id", "0"))
        frames = []
        seat_id = None

        for box_el in track_el.findall("box"):
            frame_idx = int(box_el.get("frame", "0"))
            xtl = float(box_el.get("xtl", "0"))
            ytl = float(box_el.get("ytl", "0"))
            xbr = float(box_el.get("xbr", "0"))
            ybr = float(box_el.get("ybr", "0"))

            attrs = _box_attributes(box_el)
            behavior = attrs.get("behavior", DEFAULT_BEHAVIOR)
            if seat_id is None and "seat_id" in attrs:
                seat_id = attrs["seat_id"]

            frames.append({
                "frame_idx": frame_idx,
                "bbox_xyxy": [xtl, ytl, xbr, ybr],
                "behavior": behavior,
            })

        persons.append({
            "track_id": track_id,
            "seat_id": seat_id,
            "frames": frames,
        })

    return {"persons": persons}


def _clip_duration_sec(xml_path: str | Path, fps: float) -> float:
    tree = ET.parse(str(xml_path))
    size_el = tree.getroot().find("./meta/task/size")
    n_frames = int(size_el.text) if size_el is not None and size_el.text else 0
    return round(n_frames / fps, 2) if fps > 0 else 0.0


def write_classroom_annotation(
    xml_path: str | Path,
    clip_id: str,
    split: str,
    fps: float,
    out_dir: str | Path,
) -> None:
    out_root = Path(out_dir)
    ann_dir = out_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    ann = convert_cvat_xml(xml_path)
    ann["clip_id"] = clip_id
    ann["fps"] = fps

    ann_path = ann_dir / f"{clip_id}.json"
    ann_path.write_text(json.dumps(ann, indent=2))

    meta_path = out_root / "metadata.csv"
    is_new = not meta_path.exists()
    with open(meta_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["clip_id", "split", "duration_sec", "num_persons"])
        writer.writerow([
            clip_id, split,
            _clip_duration_sec(xml_path, fps),
            len(ann["persons"]),
        ])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cvat-xml", required=True, help="Path to CVAT-for-video-1.1 XML export")
    p.add_argument("--clip-id", required=True)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--out-dir", required=True, help="ClassroomDataset root directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    write_classroom_annotation(args.cvat_xml, args.clip_id, args.split, args.fps, args.out_dir)
    print(f"Wrote {args.out_dir}/annotations/{args.clip_id}.json and appended metadata.csv")
