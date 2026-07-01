"""
test_integration.py — Integration tests across all CognEn v2.0 subsystems.

Tests cross-module interactions:
  - Feature extractor ↔ BehaviorClassifier
  - SeatMapper ↔ BehaviorClassifier (per-seat state)
  - BehaviorEvaluator ↔ BehaviorClassifier (streaming evaluation)
  - TrainingDataset base ↔ build_windows utility
  - ONNX export smoke-test (model → onnx → ort inference) if torch available

Run: pytest v2.0/tests/test_integration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


# ── Feature extractor ↔ BehaviorClassifier ───────────────────────────────────

class TestFeatureExtractorIntegration:
    def _metrics(self, **kw):
        from features.face_metrics import FaceMetricsResult
        defaults = dict(
            ear=0.30, mar=0.02, pitch=0.0, yaw=0.0, roll=0.0,
            gaze_x=0.0, gaze_y=0.0,
            blink_detected=False, blink_rate=15.0, perclos=5.0,
            fatigue_index=10.0, fidget_score=5.0, head_pose_stability=90.0,
            face_distance=0.5, face_area=0.05, mouth_openness=0.05,
            ear_variance=0.001, pose_variance=0.5,
        )
        defaults.update(kw)
        return FaceMetricsResult(**defaults)

    def test_feature_vector_shape(self):
        from training.feature_extractor import extract_feature_vector, FEATURE_DIM
        m = self._metrics()
        v = extract_feature_vector(m)
        assert v.shape == (FEATURE_DIM,)
        assert v.dtype == np.float32

    def test_feature_vector_bounded(self):
        # gaze_x/gaze_y/pitch_norm/yaw_norm (indices 0-3) are in [-1, 1] to
        # match realistic_synthetic_loader.py and
        # inference/behavior_classifier.py::_extract -- everything else
        # (EAR, PERCLOS, stability, one-hots, fidget, proximity) is [0, 1].
        # See features/feature_vector.py's module docstring for the
        # train/inference distribution-mismatch bug this fixes.
        from training.feature_extractor import extract_feature_vector
        m = self._metrics(pitch=-45.0, yaw=30.0, perclos=80.0)
        v = extract_feature_vector(m)
        assert np.all(v[:4] >= -1.0) and np.all(v[:4] <= 1.0), f"Out-of-bounds [0:4]: {v[:4]}"
        assert np.all(v[4:] >= 0.0) and np.all(v[4:] <= 1.0), f"Out-of-bounds [4:]: {v[4:]}"

    def test_phone_one_hot(self):
        from training.feature_extractor import extract_feature_vector
        class _Obj:
            cls = "phone"
        m   = self._metrics()
        v   = extract_feature_vector(m, detected_objects=[_Obj()])
        # has_phone is index 8, has_none is index 10
        assert v[8] == 1.0
        assert v[10] == 0.0

    def test_book_one_hot(self):
        from training.feature_extractor import extract_feature_vector
        class _Obj:
            cls = "book"
        m = self._metrics()
        v = extract_feature_vector(m, detected_objects=[_Obj()])
        assert v[7] == 1.0   # has_book
        assert v[10] == 0.0  # has_none

    def test_proximity_code(self):
        from training.feature_extractor import extract_feature_vector
        m    = self._metrics()
        v_ph = extract_feature_vector(m, hand_proximity="holding_phone")
        v_bk = extract_feature_vector(m, hand_proximity="holding_book")
        assert v_ph[12] == 1.0
        assert v_bk[12] == 0.75

    def test_build_input_tensor_shape(self):
        from training.feature_extractor import build_input_tensor, FEATURE_DIM
        from training.datasets.base_dataset import WINDOW_SIZE
        pytest.importorskip("torch")
        import torch
        window = np.zeros((WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)
        t = build_input_tensor(window)
        assert t.shape == (1, FEATURE_DIM, WINDOW_SIZE)


# ── Dataset base utilities ────────────────────────────────────────────────────

class TestDatasetBase:
    def test_build_windows_shape(self):
        from training.datasets.base_dataset import BaseBehaviorDataset, FEATURE_DIM, WINDOW_SIZE
        n_frames = 50
        frames = [np.zeros(FEATURE_DIM, dtype=np.float32) for _ in range(n_frames)]
        labels = list(range(n_frames))
        windows, y, groups = BaseBehaviorDataset.build_windows(
            frames, labels, window_size=WINDOW_SIZE, stride=1, group_id="clip_a")
        assert windows.shape == (n_frames - WINDOW_SIZE + 1, WINDOW_SIZE, FEATURE_DIM)
        assert y.shape == (n_frames - WINDOW_SIZE + 1,)
        assert groups.shape == y.shape
        assert np.all(groups == "clip_a")

    def test_build_windows_label_is_last_frame(self):
        from training.datasets.base_dataset import BaseBehaviorDataset, FEATURE_DIM, WINDOW_SIZE
        frames = [np.zeros(FEATURE_DIM, dtype=np.float32)] * 20
        labels = list(range(20))
        _, y, _ = BaseBehaviorDataset.build_windows(frames, labels, window_size=WINDOW_SIZE, stride=1)
        # Label of first window = label of frame at index WINDOW_SIZE-1
        assert y[0] == WINDOW_SIZE - 1

    def test_build_windows_too_short(self):
        from training.datasets.base_dataset import BaseBehaviorDataset, FEATURE_DIM, WINDOW_SIZE
        frames = [np.zeros(FEATURE_DIM, dtype=np.float32)] * (WINDOW_SIZE - 1)
        labels = list(range(WINDOW_SIZE - 1))
        windows, y, groups = BaseBehaviorDataset.build_windows(frames, labels)
        assert len(windows) == 0
        assert len(groups) == 0

    def test_encode_feature_shape(self):
        from training.datasets.base_dataset import BaseBehaviorDataset, FEATURE_DIM
        v = BaseBehaviorDataset.encode_feature()
        assert v.shape == (FEATURE_DIM,)

    def test_class_indices_stable(self):
        from training.datasets.base_dataset import BEHAVIOR_CLASSES, CLASS_TO_IDX
        for i, cls in enumerate(BEHAVIOR_CLASSES):
            assert CLASS_TO_IDX[cls] == i


# ── Streaming evaluation ──────────────────────────────────────────────────────

class TestStreamingEvaluation:
    """
    Simulate streaming behavior predictions into BehaviorEvaluator
    while a BehaviorClassifier runs live.
    """

    def test_evaluator_tracks_live_classifier(self):
        from inference.behavior_classifier import BehaviorClassifier
        from evaluation.behavior_metrics import BehaviorEvaluator
        from features.face_metrics import FaceMetricsResult
        from training.datasets.base_dataset import CLASS_TO_IDX

        clf = BehaviorClassifier()
        ev  = BehaviorEvaluator()

        m = FaceMetricsResult(
            ear=0.30, mar=0.02, pitch=-3.0, yaw=0.0, roll=0.0,
            gaze_x=0.04, gaze_y=0.0,
            blink_detected=False, blink_rate=15.0, perclos=5.0,
            fatigue_index=10.0, fidget_score=5.0, head_pose_stability=90.0,
            face_distance=0.5, face_area=0.05, mouth_openness=0.05,
            ear_variance=0.001, pose_variance=0.5,
        )
        frame = {"metrics": m, "detected_objects": [],
                 "hand_proximity": "hands_not_visible", "posture": None}

        gt_label = CLASS_TO_IDX["attentive_screen"]

        for _ in range(30):
            result = clf.update(frame)
            pred   = CLASS_TO_IDX.get(result.label, 0)
            ev.update(pred, gt_label, person_id="student_1")

        metrics = ev.compute()
        assert metrics["accuracy"] > 0.0
        assert "per_class" in metrics
        assert "temporal_consistency" in metrics


# ── Per-seat state isolation ──────────────────────────────────────────────────

class TestPerSeatStateIsolation:
    """
    Multiple BehaviorClassifiers should have independent temporal windows.
    """

    def test_independent_classifiers(self):
        from inference.behavior_classifier import BehaviorClassifier
        from features.face_metrics import FaceMetricsResult

        def _frame(pitch):
            m = FaceMetricsResult(
                ear=0.30, mar=0.02, pitch=pitch, yaw=0.0, roll=0.0,
                gaze_x=0.0, gaze_y=0.0,
                blink_detected=False, blink_rate=15.0, perclos=5.0,
                fatigue_index=10.0, fidget_score=5.0, head_pose_stability=90.0,
                face_distance=0.5, face_area=0.05, mouth_openness=0.05,
                ear_variance=0.001, pose_variance=0.5,
            )
            return {"metrics": m, "detected_objects": [],
                    "hand_proximity": "hands_not_visible", "posture": None}

        clf_screen  = BehaviorClassifier()
        clf_drowsy  = BehaviorClassifier()

        for _ in range(20):
            r_s = clf_screen.update(_frame(-3.0))     # attentive screen
            r_d = clf_drowsy.update(_frame(0.0))

        # Each classifier should converge to different labels
        # (screen stays attentive, drowsy test just checks no cross-contamination)
        assert r_s.label != "drowsy"


# ── ONNX model smoke-test ─────────────────────────────────────────────────────

class TestONNXSmoke:
    """
    If torch is available, instantiate the model and verify forward pass.
    ONNX export is only run if the model file doesn't already exist.
    """

    def test_model_forward(self):
        torch = pytest.importorskip("torch")
        from training.model import BehaviorClassifierCNN, FEATURE_DIM, WINDOW_SIZE, NUM_CLASSES
        model = BehaviorClassifierCNN()
        model.eval()
        x = torch.zeros(2, FEATURE_DIM, WINDOW_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_model_predict_proba_sums_to_1(self):
        torch = pytest.importorskip("torch")
        from training.model import BehaviorClassifierCNN, FEATURE_DIM, WINDOW_SIZE
        model = BehaviorClassifierCNN()
        model.eval()
        x = torch.randn(3, FEATURE_DIM, WINDOW_SIZE)
        probs = model.predict_proba(x)
        sums  = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_onnx_export_and_ort(self, tmp_path):
        torch   = pytest.importorskip("torch")
        ort     = pytest.importorskip("onnxruntime")
        from training.model import BehaviorClassifierCNN, FEATURE_DIM, WINDOW_SIZE, NUM_CLASSES
        import tempfile

        model = BehaviorClassifierCNN()
        model.eval()
        dummy    = torch.zeros(1, FEATURE_DIM, WINDOW_SIZE)
        onnx_out = tmp_path / "test_classifier.onnx"

        torch.onnx.export(
            model, dummy, str(onnx_out),
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        assert onnx_out.exists()

        sess  = ort.InferenceSession(str(onnx_out), providers=["CPUExecutionProvider"])
        x_np  = np.zeros((1, FEATURE_DIM, WINDOW_SIZE), dtype=np.float32)
        out   = sess.run(["logits"], {"input": x_np})[0]
        assert out.shape == (1, NUM_CLASSES)
