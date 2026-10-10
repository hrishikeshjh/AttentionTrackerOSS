#!/usr/bin/env python3
"""
ArUco head-pose ground truth for DACE objective O1 (Annex D.1, D.5).

A small printed ArUco marker is worn on a headband (marker plane roughly
parallel to the face, marker "up" = head up). For every frame of a pilot
recording this script estimates the marker pose with OpenCV and writes the
head pitch / yaw / roll in DEGREES using the DACE convention (FR1):

    +pitch = head DOWN, yaw/roll as in v1.0/src/lib/scoring/dace/head-pose-matrix.ts
    (R = Ry(yaw) . Rx(pitch) . Rz(roll) in a y-up camera frame; pitch = asin(-r12)).

The output CSV (t_ms, frame, detected, pitch_deg, yaw_deg, roll_deg,
rel_pitch_deg, reproj_px) is compared against the DACE eval log (pitchDeg /
relPitchDeg per window) to report the sign agreement (target: 100% of scripted
up/down segments) and the pitch error (target: <= 5 deg). Relative pitch is
taken against the median over the first --reference-s seconds (screen-facing
calibration), matching DACE's 5 s reference; this also cancels the constant
offset between the marker plane and the face.

Usage:
    python3 aruco_pose.py recording.mp4 --marker-mm 40 --out pose.csv \
        [--dict DICT_4X4_50] [--id 0] [--calib camera.yaml] [--reference-s 5]

Camera intrinsics: from --calib (OpenCV YAML with camera_matrix and
dist_coeffs, e.g. from a checkerboard calibration) or, if omitted, the same
approximation as DACE's solvePnP baseline (f = frame width, principal point =
frame centre, no distortion). Use a real calibration for reported numbers.

Requires: opencv-contrib-python >= 4.7 (cv2.aruco), numpy. Only video files
on the local machine are read; nothing is uploaded.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - reported at runtime
    cv2 = None
    np = None


def load_intrinsics(path: str | None, width: int, height: int):
    if path:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        k = fs.getNode("camera_matrix").mat()
        d = fs.getNode("dist_coeffs").mat()
        fs.release()
        if k is None:
            raise ValueError(f"{path}: no camera_matrix")
        return k.astype(np.float64), (d if d is not None else np.zeros(5)).astype(np.float64)
    k = np.array([[width, 0, width / 2.0], [0, width, height / 2.0], [0, 0, 1]], dtype=np.float64)
    return k, np.zeros(5, dtype=np.float64)


def euler_dace(r_cv) -> tuple[float, float, float]:
    """OpenCV-frame rotation (x right, y down, z forward) -> DACE Euler degrees.

    ArUco marker axes: x right, y up (in the marker plane), z out of the
    marker toward the camera. r_cv maps marker axes into the OpenCV camera
    frame; a frontal upright marker gives r_cv = diag(1, -1, -1). Moving the
    result into the y-up, z-toward-viewer frame of head-pose-matrix.ts,
    R = F r_cv with F = diag(1, -1, -1), gives R = I for a frontal marker, and
    a forward nod (face normal tilting toward the camera's +y_cv, i.e. down)
    gives R[1][2] = -sin(phi), so pitch = asin(-R[1][2]) = +phi (head down).
    """
    f = np.diag([1.0, -1.0, -1.0])
    r = f @ r_cv
    clamp = max(-1.0, min(1.0, -r[1, 2]))
    pitch = math.degrees(math.asin(clamp))
    yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
    roll = math.degrees(math.atan2(r[1, 0], r[1, 1]))
    return pitch, yaw, roll


def marker_object_points(size_mm: float):
    h = size_mm / 2.0
    # cv2.aruco corner order: top-left, top-right, bottom-right, bottom-left; marker y up.
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def make_detector(dict_name: str):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    if hasattr(aruco, "ArucoDetector"):
        det = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        return lambda gray: det.detectMarkers(gray)
    params = aruco.DetectorParameters_create()
    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--marker-mm", type=float, required=True, help="printed marker side length (mm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--id", type=int, default=None, help="marker id to track (default: first detected)")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--reference-s", type=float, default=5.0)
    ap.add_argument("--pitch-sign", type=float, default=1.0, help="sign correction if the marker is mounted upside down")
    a = ap.parse_args(argv)

    if cv2 is None or not hasattr(cv2, "aruco"):
        print("opencv-contrib-python (cv2.aruco) and numpy are required", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        print(f"cannot open {a.video}", file=sys.stderr)
        return 2
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k, dist = load_intrinsics(a.calib, width, height)
    obj = marker_object_points(a.marker_mm)
    detect = make_detector(a.dict)

    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect(gray)
        row = {"t_ms": round(t_ms, 1), "frame": frame_idx, "detected": 0,
               "pitch_deg": "", "yaw_deg": "", "roll_deg": "", "rel_pitch_deg": "", "reproj_px": ""}
        if ids is not None and len(ids):
            ids = ids.flatten().tolist()
            j = ids.index(a.id) if (a.id is not None and a.id in ids) else (0 if a.id is None else None)
            if j is not None:
                img = corners[j].reshape(4, 2).astype(np.float64)
                ok_pnp, rvec, tvec = cv2.solvePnP(obj, img, k, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok_pnp:
                    r_cv, _ = cv2.Rodrigues(rvec)
                    pitch, yaw, roll = euler_dace(r_cv)
                    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
                    err = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img) ** 2, axis=1))))
                    row.update(detected=1, pitch_deg=round(a.pitch_sign * pitch, 3), yaw_deg=round(yaw, 3),
                               roll_deg=round(roll, 3), reproj_px=round(err, 3))
        rows.append(row)
        frame_idx += 1
    cap.release()

    ref_vals = [r["pitch_deg"] for r in rows if r["detected"] and r["t_ms"] <= a.reference_s * 1000]
    ref = statistics.median(ref_vals) if ref_vals else None
    for r in rows:
        if r["detected"] and ref is not None:
            r["rel_pitch_deg"] = round(r["pitch_deg"] - ref, 3)

    out = Path(a.out)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["t_ms", "frame", "detected", "pitch_deg", "yaw_deg", "roll_deg", "rel_pitch_deg", "reproj_px"])
        w.writeheader()
        w.writerows(rows)
    det = sum(r["detected"] for r in rows)
    print(f"{len(rows)} frames, marker detected in {det}; reference pitch "
          f"{'n/a' if ref is None else f'{ref:.2f} deg'} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
