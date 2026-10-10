# Synthetic Generator Calibration

`realistic_synthetic_loader.py`'s `_SPECS` table (per-class means/std-devs
for gaze/pitch/yaw/EAR/PERCLOS/stability/fidget) was constructed from
biometric literature and reasoning about each behavior class, but has
**never been validated against real extracted distributions** — no real
footage has been run through this pipeline yet. This document is the
template for that validation once real data exists; it is currently
empty of results, not because the check was done and passed, but because
it hasn't been run.

## What to do once real classroom/DAiSEE/EngageNet footage is available

1. Run the real footage through `features/feature_vector.py::extract_feature_vector_from_landmarks`
   (and its 15-dim twin `extract_feature_vector_v2_from_landmarks`; the same canonical
   extraction used by `daisee_loader.py` / `engagenet_loader.py` /
   `classroom_loader.py`, via `TrackSequenceEncoder` at 5 Hz) to get real per-frame
   feature vectors, labeled by behavior class via the dataset's own labels.
2. For each of the 6 classes and each continuous feature (gaze_x, gaze_y,
   pitch_norm, yaw_norm, ear, perclos, stability, fidget), compute the real
   sample mean and std-dev, restricted to frames of that class.
3. Compare against the corresponding `_SPECS[class_name]` entry in
   `realistic_synthetic_loader.py` (reproduced below for reference).
4. For any feature/class pair with a large mismatch (recommend a
   Kolmogorov-Smirnov two-sample test, or a simple visual histogram
   overlay for a faster first pass), consider whether:
   - The synthetic constant should be adjusted to match reality, or
   - The mismatch reveals a real difference between the synthetic
     class definition and the actual behavior (e.g. "attentive_reading"
     pitch might genuinely vary more across a real, diverse student
     population than the synthetic generator's `pitch_sd=0.05` assumes).
5. Record the comparison result (pass/fail per feature/class, and the
   real vs. synthetic numbers) in this file, replacing this section.

## Current synthetic constants (unvalidated against real data)

Reproduced from `realistic_synthetic_loader.py::_SPECS` for convenience —
this is the SOURCE OF TRUTH copy in that file; if the two ever disagree,
the code is authoritative and this table is stale.

| Class | pitch_mu (/90, relative; down = negative) | gaze_y_mu | ear_mu | perc_mu | stab_mu | fidget_mu |
|---|---|---|---|---|---|---|
| attentive_screen  |  0.00 | 0.00 | 0.32 | 0.03 | 0.87 | 0.10 |
| attentive_reading | -0.30 | -0.55 | 0.30 | 0.04 | 0.82 | 0.08 |
| using_phone       | -0.36 | -0.60 | 0.30 | 0.04 | 0.76 | 0.28 |
| distracted        |  0.03 | 0.00 | 0.31 | 0.05 | 0.48 | 0.38 |
| drowsy            | -0.08 | 0.02 | 0.18 | 0.32 | 0.58 | 0.04 |
| absent            |  0.00 | 0.00 | 0.30 | 0.00 | 0.00 | 0.00 |

Notable design assumptions worth checking first against real data (most
likely to be wrong, since they weren't derived from a specific published
source at construction time):
- `attentive_reading` vs `using_phone` are deliberately very close in
  pitch (-0.30 vs -0.36) and gaze_y (-0.55 vs -0.60; down = negative, pitch is relative to the per-track screen-facing reference) — this is intentional
  (they're supposed to be the hard, visually-similar pair the Context-Gate
  Override exists to disambiguate), but the exact gap size (0.06 in pitch/90 units ≈ 5°)
  is a guess, not measured.
- `drowsy`'s `perc_mu=0.32` (32% PERCLOS) and `ear_mu=0.18` are set to sit
  clearly past the EAR<0.22/PERCLOS>20% thresholds used elsewhere in the
  codebase (e.g. `inference/behavior_classifier.py`) — internally
  consistent with those thresholds, but the thresholds themselves are also
  unvalidated against real drowsy-student data.
- `distracted`'s wide gaze/yaw variance (`gaze_x_sd=0.32`, `yaw_sd=0.20`)
  assumes distraction looks like "looking around" rather than e.g. "staring
  blankly at a fixed off-screen point," which would look very different in
  the feature space (low variance, off-center mean) and isn't represented
  as a distinct sub-case at all currently.

## Result log

_(empty — no real-data comparison has been run yet)_
