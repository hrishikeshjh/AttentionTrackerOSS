# behavior_classifier_v3 — synthetic-only note (2026-10-10)

**Read first: every number below is on synthetic data drawn from the same
hand-parameterised generator the model was trained on. It is circular. It says
nothing about accuracy on real classroom video. No real-data evaluation of any
CognEn behavior model exists.**

## What changed versus v2

| | v2 (`behavior_classifier_v2.onnx`) | v3 (`behavior_classifier_v3.onnx`) |
|---|---|---|
| Input | (B, 13, 15) | (B, 15, 15): FEATURE_VECTOR_VERSION 2 = the 13 v1.0-compatible channels + `posture_upright`, `posture_lean_norm` |
| Sign convention | pitch mean negative but gaze_y **positive** for reading/phone (inconsistent); server pitch came from the legacy formula, which rises when looking down | pitch_norm and gaze_y both **down = negative** (features/feature_vector.py); server measures them with the DACE port (features/dace_measurement.py) |
| Pitch meaning | absolute legacy units | degrees relative to a per-track screen-facing reference |
| Generator | objects always held | objects sometimes present but not held (p_hold), phone/book on the desk in screen / distracted windows, 15% windows with unknown posture |

The server default (`server/config.py BEHAVIOR_CLASSIFIER_MODEL`) now points
at v3. v1 and v2 models are kept; the classifier feeds 13-channel models the
first 13 channels.

## Training (CPU, ~90 s)

```
cd v2.0
python3 training/train_classifier.py --realistic --realistic-n-per-class 2000 \
    --realistic-augment 3 --penalty-loss --epochs 40 --patience 8 --batch 256 \
    --seed 42 --realistic-seed 42 --output models/behavior_classifier_v3.pt
```

40,800 train windows / 1,800 val windows (6 classes, balanced), 20,742
parameters, ConfusionPenaltyLoss. Best checkpoint epoch 4 (val acc 1.000,
val CE 0.040; checkpoint selection now breaks accuracy ties by val CE). Early
stop at epoch 12. Exported as a single-file ONNX, opset 17, TorchScript
exporter (`dynamo=False`), input `input` (batch, 15, 15), output `logits`.

## Independent-seed synthetic hold-out

```
python3 evaluation/synthetic_holdout_eval.py --model models/behavior_classifier_v3.onnx \
    --seed 2026 --n-per-class 1000 --out evaluation/reports/synthetic_holdout_v3.json
```

| Set | n | Accuracy | Macro-F1 |
|---|---|---|---|
| clean (seed 2026) | 6000 | 1.000 | 1.000 |
| augmented (time warp, dropout, gaze jitter, context injection; seed 2027) | 6000 | 1.000 | 1.000 |
| reading/phone windows with pitch and gaze_y negated (old +down convention) | 2000 | recall 0.000 | — |

Ablations on another seed (99, 500/class): removing all object context from
reading/phone windows still gives recall 1.000; forcing unknown posture
still gives accuracy 1.000; screen windows with pitch set to 25° down, eyes
down and a book become `attentive_reading` (500/500).

## How to read this

- Perfect scores mean the generator's classes are trivially separable (e.g.
  fidget / stability / lean differ by class), not that the model works. The
  object-context ablation shows the model does **not** need the phone/book
  channels on synthetic data, which will not hold on real video.
- The sign-flip result is the one useful finding: the model depends on the
  down = negative convention, so feeding it the legacy (+down, uncorrected)
  pitch would break it. This is why v1.0 should load it only in DACE mode
  (`loadModel(url, { featureDim: 15 })`), where the legacy fields carry the
  negated DACE values.
- Contract checks: tests/test_cross_language_contract.py,
  tests/test_feature_vector_v2_contract.py (v3 has 15 input channels; server
  default points at it).
- Still needed: real data (DAiSEE / EngageNet / own classroom recordings with
  the v2 measurement pipeline), subject-wise splits, and calibration of the
  DACE thresholds (θ_down, γ_down, k) on pilot recordings.

## Addendum (2026-10-10, later): training split leak, re-checked

- **The shipped v3 checkpoint was trained with a leaky validation split.**
  `train_classifier.py` (the version that trained v3) pooled the synthetic train and val windows and
  re-split them with `GroupShuffleSplit`, while synthetic windows had no group
  id, so augmented copies of training windows could land in validation. The
  "40,800 train / 1,800 val" counts above are the loader's sizes, not the
  split that was actually used; the "val acc 1.000" is therefore not a clean
  validation number.
- Fixed: every synthetic base window has a group id shared by its augmented
  copies, the loaders' own train/val splits are kept, and the split is
  asserted group-disjoint (`assert_no_group_leakage`). Model selection now
  uses val macro-F1 (ties by val CE) instead of accuracy, and `export_onnx`
  checks ONNX/PyTorch logit parity.
- Re-check (not shipped): the documented command above, run with the fixed
  split (40,800 train / 1,800 val, 10,200 / 1,800 groups, 0 shared), selects
  epoch 1 (val macro-F1 1.000, val CE 0.036) and gives the **same**
  independent-seed hold-out results as v3: clean 1.000 / 1.000, augmented
  1.000 / 1.000, sign-flipped reading/phone recall 0.000. ONNX parity
  1.4e-6. The leak did not change anything measurable on this (trivially
  separable) generator, so `behavior_classifier_v3.*` were **not** replaced.
  The shipped v3 ONNX matches its checkpoint (parity < 1e-4,
  tests/test_training_export.py).
- Everything above is still synthetic and circular. The earlier
  `eval_20260309_*.json` / `experiment_report*.md` are RETRACTED and moved to
  `archive/` (see `archive/README.md`); this note is the only current report.
