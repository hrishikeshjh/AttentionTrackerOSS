# RETRACTED — archived evaluation outputs (2026-03-09)

**Nothing in this folder is a valid measurement. Do not quote any number from
these files.** They are kept only as a record of what was previously
committed.

| File | Why it is retracted |
|---|---|
| `eval_20260309_125702.json` | Written by the pre-fix `evaluate.py`. Verified against the runner in git history (commit 37c9ec3, `evaluation/evaluate.py` lines ~75-157): its behavior section was a synthetic "80 % correct + 20 % noise" prediction generator, not a model; tracking ran ground truth through a fixed-offset noise injector; occupancy was a synthetic ~95 %-accuracy generator; latency timed only the 13x15 CNN (~0.02 ms). The "8/11 targets met" summary is fabricated. Temporal consistency was a vacuous 1.0 (no person id was ever passed); MOTA / IDF1 = 1.0 on the noise-injected ground truth; the occupancy section's 1,763 "absence events" all have 0 s latency (the re-recording bug); latency p95 0.02 ms = CNN only. |
| `eval_20260309_131330.json` | Same pre-fix runner and same synthetic sections ("formal runner, same 8/11"). |
| `experiment_report.md` | Experiment 1: a model trained on 1,000 random uniform windows (val acc 0.213 = noise), evaluated with the fabricated runner above. |
| `experiment_report_v2.md` | Experiment 2: trained and tested on the same hand-written synthetic generator (different seeds) -> the 1.000 scores are circular. Its training split also leaked: synthetic windows had no group id, so augmented copies of training windows could land in validation. |

Other defects of that pipeline, fixed since (see the code comments in
`training/` and `evaluation/`): "IDF1" was per-frame detection F1; MOTA was
clipped to [0, 1]; tracking evaluation used every annotation instead of the
test split; the occupancy latency re-recorded a 0 s event every frame;
loaders fed `np.zeros` for no-face frames, constant PERCLOS / stability /
fidget, and native-fps frames (15 frames = 0.5 s instead of 3 s); DAiSEE was
grouped by clip and EngageNet by row offset (participants leaked across the
split).

**Current report:** `../behavior_classifier_v3_synthetic_note.md` (synthetic
only, explicitly circular). New `evaluate.py` reports carry
`data_source` (`real` / `synthetic` / `none`) and `claims_allowed`; no real-data
evaluation of any CognEn model exists yet.
