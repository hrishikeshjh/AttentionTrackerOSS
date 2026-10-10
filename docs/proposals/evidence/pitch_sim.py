#!/usr/bin/env python3
"""Reproduce the legacy head-pitch diagnosis (DACE proposal section 1, annex E).

Usage (from the repository root):
    python3 docs/proposals/evidence/pitch_sim.py

Applies the EXACT legacy pitch formula and constants of
v1.0/src/lib/scoring/ to the standard OpenCV 3D head model under known
rotations and prints the table in proposal section 1.

The constants are NOT copied here: they are read from
legacy_pitch_constants.json, which is generated from the TypeScript source by
v1.0/src/lib/scoring/__tests__/pitch-characterisation.test.ts (run
`npx vitest run` in v1.0/ to refresh it). Requires numpy; uses OpenCV's
projectPoints when cv2 is installed, otherwise an equivalent pinhole
projection implemented below.

Values are simulated, not measured.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

try:  # optional
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 is optional
    cv2 = None

HERE = Path(__file__).resolve().parent
CONSTANTS = HERE / "legacy_pitch_constants.json"


def load_constants() -> dict:
    if not CONSTANTS.exists():
        sys.exit(f"missing {CONSTANTS}; run `npx vitest run` in v1.0/ to generate it")
    return json.loads(CONSTANTS.read_text())


def rotation_head_down(deg: float) -> np.ndarray:
    """Rotation about the camera x-axis in the OpenCV frame (x right, y down, z forward).
    Positive angle = head tilts DOWN (face normal turns toward +y)."""
    a = math.radians(deg)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]]
    )


def model_points(c: dict) -> dict[str, np.ndarray]:
    # OpenCV head model is given with y up and z toward the viewer; convert to
    # the camera convention (x, -y, -z) so that a frontal face is R = I.
    return {k: np.array([v[0], -v[1], -v[2]], dtype=float) for k, v in c["model3dMm"].items()}


def project(
    points: dict[str, np.ndarray],
    R: np.ndarray,
    distance_mm: float | None,
    f: float = 640.0,
    face_above_axis_mm: float = 0.0,
) -> dict[str, np.ndarray]:
    """Rotate about the nose tip, then project. distance_mm=None -> orthographic.
    face_above_axis_mm > 0 places the face above the optical axis (camera below
    eye level, as with a laptop webcam)."""
    names = list(points)
    P = np.stack([points[n] for n in names])  # rotation about the nose tip (origin)
    if distance_mm is None:
        Q = P @ R.T
        uv = Q[:, :2] * 0.2 + np.array([320.0, 240.0])
    elif cv2 is not None:
        rvec, _ = cv2.Rodrigues(R)
        K = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]])
        uv, _ = cv2.projectPoints(P, rvec, np.array([0.0, -face_above_axis_mm, distance_mm]), K, None)
        uv = uv.reshape(-1, 2)
    else:
        Q = P @ R.T + np.array([0.0, -face_above_axis_mm, distance_mm])
        uv = f * Q[:, :2] / Q[:, 2:3] + np.array([320.0, 240.0])
    return dict(zip(names, uv))


def legacy_pitch(img: dict[str, np.ndarray], c: dict) -> float:
    """pitch = (nose.y - eyeMid.y) / interocular * PITCH_GAIN  (head-pose.ts)."""
    nose = img["noseTip"]
    le = img["leftEyeLeftCorner"]
    re = img["rightEyeRightCorner"]
    mid = (le + re) / 2.0
    eye_dist = float(np.hypot(*(le - re)))
    return float((nose[1] - mid[1]) / eye_dist * c["PITCH_GAIN"])


def pitch_sub_score(v: float, c: dict) -> float:
    s = c["pitchSubScore"]
    k = s["constants"]
    m = s["maxDeviation"]
    a = abs(v)
    if a <= m:
        return k["top"] - a / m * k["insideDrop"]
    return max(k["floor"], k["outsideStart"] - (a - m) * k["outsideSlope"])


def describe(deg: int) -> str:
    if deg == 0:
        return "Level"
    return f"{abs(deg)}° {'down' if deg > 0 else 'up'}"


def table(c: dict, distance_mm: float | None, above_mm: float = 0.0) -> tuple[list[float], list[float]]:
    pts = model_points(c)
    pitches, scores = [], []
    for deg in c["tiltsDeg"]:
        v = legacy_pitch(project(pts, rotation_head_down(deg), distance_mm, face_above_axis_mm=above_mm), c)
        pitches.append(v)
        scores.append(pitch_sub_score(v, c))
    return pitches, scores


def print_table(title: str, c: dict, pitches: list[float], scores: list[float]) -> None:
    heads = [describe(d) for d in c["tiltsDeg"]]
    print(title)
    print(f"  {'True head tilt':<18}" + "".join(f"{h:>12}" for h in heads))
    print(f"  {'CognEn pitch':<18}" + "".join(f"{p:>+12.1f}" for p in pitches))
    print(f"  {'Pitch sub-score':<18}" + "".join(f"{s:>12.1f}" for s in scores))
    print()


def main() -> int:
    c = load_constants()
    print(f"Legacy formula: {c['formula']}  (PITCH_GAIN={c['PITCH_GAIN']})")
    print(f"Constants from: {CONSTANTS.name} (generated from {', '.join(c['sources'][:2])})")
    print(f"Projection backend: {'OpenCV projectPoints' if cv2 is not None else 'numpy pinhole (cv2 not installed)'}\n")

    p, s = table(c, None)
    print_table("Orthographic / distant camera (proposal section 1):", c, p, s)

    want = c["proposalTable"]
    ok = all(abs(round(a, 1) - b) <= 0.051 for a, b in zip(p, want["pitch"])) and all(
        abs(round(a, 1) - b) <= 0.051 for a, b in zip(s, want["subScore"])
    )
    ts = c.get("typescriptResults", [])
    ts_ok = all(abs(r["legacyPitch"] - a) < 1e-4 for r, a in zip(ts, p)) if ts else False

    # Sensitivity: with the camera level with the face, perspective cancels in
    # the ratio (all three points share one depth scale). An off-axis laptop
    # camera changes the numbers but not the conclusion.
    for d, above in ((500.0, 0.0), (500.0, 150.0), (700.0, 250.0)):
        pp, ss = table(c, d, above)
        print_table(
            f"Pinhole camera, f = 640 px, distance {d:.0f} mm, face {above:.0f} mm above the optical axis (sensitivity):",
            c, pp, ss,
        )

    print(f"Proposal table reproduced (1 decimal): {'YES' if ok else 'NO'}")
    print(f"Matches the TypeScript characterisation test: {'YES' if ts_ok else 'NO'}")
    down = [v for v, d in zip(p, c["tiltsDeg"]) if d > 0]
    print(
        f"Reading/phone branch (pitch < {c['contextAwarePitchDownThreshold']}) reachable by tilting down: "
        f"{'YES' if any(v < c['contextAwarePitchDownThreshold'] for v in down) else 'NO'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
