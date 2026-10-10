"""
Training-data pipeline: leakage-free grouping / splitting, loader == inference
feature encoding, 5 Hz cadence, grounded classes. Fake videos and fake face
landmarkers only (no dataset downloads); cv2 tests are skipped without cv2.

Run: pytest v2.0/tests/test_training_data_pipeline.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))

from features.dace_measurement import FaceObservation, matrix_from_euler  # noqa: E402
from features.feature_vector import encode_frame_v2  # noqa: E402
from inference.behavior_classifier import ABSENT_MIN_NO_FACE_FRAMES, BehaviorClassifier  # noqa: E402
from perception.face_landmarks import Point2D  # noqa: E402
from training.datasets.base_dataset import (  # noqa: E402
    BEHAVIOR_CLASSES,
    CLASS_TO_IDX,
    ENGAGEMENT_PROXY_CLASSES,
    FEATURE_DIM,
    WINDOW_SIZE,
    BaseBehaviorDataset,
)
from training.datasets.realistic_synthetic_loader import RealisticSyntheticDataset  # noqa: E402
from training.datasets.sequence_encoder import (  # noqa: E402
    LabelledSequence,
    TrackSequenceEncoder,
    segment_sequence,
    split_on_gaps,
    subsample_by_timestamp,
)
from training.train_classifier import (  # noqa: E402
    GroupLeakageError,
    _assemble_split,
    _train_val_split,
    assert_no_group_leakage,
    class_weights_np,
    is_better,
    macro_f1,
)

# ── Fakes ─────────────────────────────────────────────────────────────────────


def _face_68(dx=0.0, dy=0.0, eye_open=1.0):
    pts = [Point2D(50.0 + dx, 50.0 + dy) for _ in range(68)]
    h = 3.0 * eye_open
    eye = {36: (20, 40), 37: (25, 40 - h), 38: (30, 40 - h), 39: (35, 40), 40: (30, 40 + h), 41: (25, 40 + h),
           42: (65, 40), 43: (70, 40 - h), 44: (75, 40 - h), 45: (80, 40), 46: (75, 40 + h), 47: (70, 40 + h),
           30: (50, 55), 48: (35, 70), 54: (65, 70), 51: (50, 66), 57: (50, 76)}
    for i, (x, y) in eye.items():
        pts[i] = Point2D(float(x) + dx, float(y) + dy)
    return pts


def _bs(down=0.0):
    keys = ("eyeLookDownLeft", "eyeLookDownRight", "eyeLookUpLeft", "eyeLookUpRight",
            "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight")
    d = {k: 0.0 for k in keys}
    d["eyeLookDownLeft"] = d["eyeLookDownRight"] = down
    return d


def _obs(pitch=0.0, yaw=0.0, down=0.0, dx=0.0):
    return FaceObservation(landmarks68=_face_68(dx), mesh=None,
                           matrix=matrix_from_euler(pitch, yaw, 0.0), blendshapes=_bs(down))


class _ColorFaceDet:
    """Face iff the crop's mean green > 64; head pitch (deg, +down) = (blue - 128) / 4."""
    available = True
    status = "ready"

    def __init__(self):
        self.calls = 0

    def detect_full(self, crop, offset=(0, 0)):
        self.calls += 1
        g = float(crop[..., 1].mean())
        if g <= 64:
            return []
        pitch = (float(crop[..., 0].mean()) - 128.0) / 4.0
        return [_obs(pitch=pitch, yaw=0.0, dx=offset[0])]

    def detect(self, crop, offset=(0, 0)):
        return [o.landmarks68 for o in self.detect_full(crop, offset)]


# ── Cadence ───────────────────────────────────────────────────────────────────

def test_subsample_30fps_to_5hz_gives_3s_windows():
    ts = [i * 1000.0 / 30.0 for i in range(30 * 6)]          # 6 s at 30 fps
    keep = subsample_by_timestamp(ts)
    assert keep[:4] == [0, 6, 12, 18]
    kept_ts = np.array([ts[i] for i in keep])
    assert np.allclose(np.diff(kept_ts), 200.0, atol=1.0)
    # 15 kept frames span 14 periods = 2.8 s (+ one frame) = the server's 3 s window
    assert kept_ts[14] - kept_ts[0] == pytest.approx(2800.0, abs=1.0)
    assert len(keep) == 30


@pytest.mark.parametrize("fps,expect_step", [(25.0, 5), (15.0, 3), (10.0, 2), (5.0, 1)])
def test_subsample_other_rates(fps, expect_step):
    ts = [i * 1000.0 / fps for i in range(int(fps * 4))]
    keep = subsample_by_timestamp(ts)
    assert np.all(np.diff(keep) == expect_step)


def test_subsample_slow_source_keeps_every_frame_and_rejects_unsorted():
    assert subsample_by_timestamp([0, 300, 650, 1000]) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        subsample_by_timestamp([0, 400, 200])


def test_split_on_gaps():
    ts = [0, 200, 400, 1400, 1600]
    assert [(s.start, s.stop) for s in split_on_gaps(ts)] == [(0, 3), (3, 5)]


# ── Encoder == inference ─────────────────────────────────────────────────────

def test_no_face_frame_uses_inference_defaults_not_zeros():
    enc = TrackSequenceEncoder()
    f = enc.step(0, None, face_detector_available=True)
    assert f.no_face is True
    assert not np.allclose(f.vector, 0.0)
    # identical to what BehaviorClassifier._extract feeds the model
    clf = BehaviorClassifier(session=None)
    ref = np.array(clf._extract({"metrics": f.metrics, "detected_objects": [],
                                 "hand_proximity": "hands_not_visible", "posture": None}), dtype=np.float32)
    assert np.array_equal(f.vector, ref)
    assert f.vector[4] == pytest.approx(0.3)     # EAR default
    assert f.vector[10] == 1.0                    # has_none
    assert tuple(f.vector[13:]) == (1.0, 0.0)     # posture unknown


def test_perclos_stability_fidget_are_computed_from_the_sequence():
    """A still head vs a nodding head give different stability / fidget, and
    eyes closed for seconds raise PERCLOS -- none of these are constants."""
    still, nod = TrackSequenceEncoder(), TrackSequenceEncoder()
    for i in range(60):
        t = i * 200
        a = still.step(t, _obs(pitch=0.0))
        b = nod.step(t, _obs(pitch=25.0 * np.sin(i * 1.3), yaw=20.0 * np.cos(i * 0.9)))
    assert a.vector[6] > b.vector[6]          # stability
    assert b.vector[11] > a.vector[11]        # fidget
    assert len({round(float(v), 4) for v in (a.vector[6], b.vector[6])}) == 2

    closed = TrackSequenceEncoder()
    vals = []
    for i in range(120):
        obs = FaceObservation(_face_68(eye_open=0.05 if i >= 60 else 1.0), None, matrix_from_euler(0, 0, 0), _bs())
        vals.append(closed.step(i * 200, obs).vector[5])
    assert vals[-1] > vals[55]                # PERCLOS rises after the eyes close


def test_loader_encoder_matches_room_analyzer_per_person_path():
    """Same observation sequence -> RoomAnalyzer._analyze_person (server) and
    TrackSequenceEncoder (loaders) produce identical feature vectors."""
    from inference import room_analyzer as ra
    from tracking.byte_tracker import Track

    seq = [_obs(pitch=p, down=d) for p, d in
           [(0, 0), (2, 0), (-1, 0), (25, 0.7), (30, 0.8), (0, 0), (5, 0.1)] * 4] + [None, None, _obs(3, 0)]

    class _SeqDet:
        available = True
        status = "ready"

        def __init__(self):
            self.i = 0

        def detect_full(self, crop, offset=(0, 0)):
            o = seq[self.i]
            self.i += 1
            return [o] if o is not None else []

    class _E:
        def detect(self, f):
            return []

    class _T:
        def update(self, dets, frame):
            return [Track(7, (100, 50, 300, 450), 0.9)]

    class _S:
        def assign(self, tracks):
            return []

    a = ra.RoomAnalyzer("r", {}, person_detector=_E(), object_detector=_E(), pose_estimator=_E(),
                        face_landmarks=_SeqDet(), tracker=_T(), seat_mapper=_S(),
                        behavior_session=None, analyze_posture=lambda p: None, hand_proximity_source="off")
    enc = TrackSequenceEncoder()
    frame = np.zeros((ra.INPUT_HEIGHT, ra.INPUT_WIDTH, 3), dtype=np.uint8)
    for i, o in enumerate(seq):
        ts = 10_000 + i * 200
        state = a._person_states.get(7) or a._new_person_state()
        a._person_states[7] = state
        a._analyze_person(frame, Track(7, (100, 50, 300, 450), 0.9), [], [], state, ts)
        server_vec = np.array(state.behavior_clf._buffer[-1], dtype=np.float32)
        area = (200 * 400) / (ra.INPUT_WIDTH * ra.INPUT_HEIGHT)
        mine = enc.step(ts, o, face_detector_available=True, face_area=area)
        assert np.array_equal(server_vec, mine.vector), i
        assert state.last_result is not None
        assert state.last_result.label == mine.result.label


# ── Grouping / leakage ───────────────────────────────────────────────────────

def test_synthetic_augmented_copies_share_their_base_group():
    tr = RealisticSyntheticDataset(split="train", n_per_class=20, augment_factor=3, seed=5)
    va = RealisticSyntheticDataset(split="val", n_per_class=20, augment_factor=0, seed=5)
    g_tr = [g for _, _, g in tr]
    g_va = [g for _, _, g in va]
    assert None not in g_tr and None not in g_va
    # every train group = 1 base window + 3 augmented copies
    counts = {g: g_tr.count(g) for g in set(g_tr)}
    assert set(counts.values()) == {4}
    assert set(g_tr).isdisjoint(g_va)


def test_assemble_split_keeps_loader_splits_and_asserts_no_leakage():
    X = np.zeros((10, WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)
    y = np.array([0, 3, 4, 0, 3, 4, 0, 3, 4, 0])
    groups = np.array(["daisee:A"] * 3 + ["daisee:B"] * 3 + ["daisee:A", "daisee:C", "daisee:C", "daisee:D"],
                      dtype=object)
    splits = np.array(["train"] * 6 + ["val"] * 4, dtype=object)
    tr, va, g_tr, g_va, info = _assemble_split(X, y, groups, splits)
    assert list(tr) == [0, 1, 2, 3, 4, 5]                 # loader train split kept as is
    assert list(va) == [7, 8, 9]                          # subject A removed from val (also in train)
    assert info["val_groups_removed_for_leakage"] == 1
    assert set(g_tr.tolist()).isdisjoint(g_va.tolist())


def test_untagged_windows_are_group_split():
    rng = np.random.default_rng(0)
    groups = np.array([f"g{i // 5}" for i in range(100)], dtype=object)
    X = rng.random((100, WINDOW_SIZE, FEATURE_DIM)).astype(np.float32)
    y = rng.integers(0, 6, 100)
    tr, va, g_tr, g_va, _ = _assemble_split(X, y, groups, np.array([None] * 100, dtype=object), 0.2, 3)
    assert len(va) > 0 and set(g_tr.tolist()).isdisjoint(g_va.tolist())
    Xtr, _, Xva, _ = _train_val_split(X, y, groups, 0.2, 3)
    assert len(Xtr) + len(Xva) == 100


def test_leakage_assertion():
    assert_no_group_leakage(["a", "b"], ["c"])
    with pytest.raises(GroupLeakageError):
        assert_no_group_leakage(["a", "b"], ["b", "c"])


def test_daisee_subject_parsing():
    from training.datasets.daisee_loader import daisee_subject_id
    split = Path("/d/DataSet/Train")
    assert daisee_subject_id("1100011002.avi") == "110001"
    assert daisee_subject_id("1100011002", split / "110001" / "1100011002" / "1100011002.avi", split) == "110001"
    with pytest.raises(ValueError):
        daisee_subject_id("clipA")


def test_engagenet_subject_parsing():
    from training.datasets.engagenet_loader import engagenet_subject_id
    assert engagenet_subject_id("x.jpg", "S7") == "S7"
    assert engagenet_subject_id("S012/clip3/0001.jpg") == "S012"
    assert engagenet_subject_id("S012_0001.jpg") == "S012"
    with pytest.raises(ValueError):
        engagenet_subject_id("0001.jpg")


# ── Grounding ─────────────────────────────────────────────────────────────────

class _OneSeq(BaseBehaviorDataset):
    DATASET_NAME = "t"
    GROUNDED_CLASSES = ENGAGEMENT_PROXY_CLASSES

    def __init__(self, seq):
        super().__init__("", "train")
        self.stride = 1
        self._seq = seq

    def iter_sequences(self):
        yield self._seq


def _seq_with(no_face_frames: int, label: str):
    enc = TrackSequenceEncoder()
    s = LabelledSequence("t:1", "t:1")
    for i in range(WINDOW_SIZE):
        s.frames.append(enc.step(i * 200, None if i < no_face_frames else _obs()))
        s.labels.append(CLASS_TO_IDX[label])
    return s


def test_engagement_dataset_drops_absent_like_windows():
    ds = _OneSeq(_seq_with(ABSENT_MIN_NO_FACE_FRAMES, "attentive_screen"))
    assert list(ds) == [] and ds.dropped_absent_windows == 1
    ds = _OneSeq(_seq_with(ABSENT_MIN_NO_FACE_FRAMES - 1, "attentive_screen"))
    assert len(list(ds)) == 1


def test_engagement_dataset_refuses_ungroundable_classes():
    ds = _OneSeq(_seq_with(0, "using_phone"))
    with pytest.raises(ValueError, match="cannot ground"):
        list(ds)


def test_grounded_sets():
    from training.datasets.classroom_loader import ClassroomDataset
    from training.datasets.daisee_loader import DAiSEEDataset
    from training.datasets.engagenet_loader import EngageNetDataset
    assert DAiSEEDataset.GROUNDED_CLASSES == EngageNetDataset.GROUNDED_CLASSES == \
        frozenset({"attentive_screen", "distracted", "drowsy"})
    assert ClassroomDataset.GROUNDED_CLASSES == frozenset(BEHAVIOR_CLASSES)


# ── Training helpers ──────────────────────────────────────────────────────────

def test_class_weights_ignore_absent_classes():
    y = np.array([0] * 90 + [3] * 10)
    w = class_weights_np(y)
    assert w[1] == w[2] == w[4] == w[5] == 0.0
    assert w[3] == pytest.approx(9 * w[0])
    assert (w[0] + w[3]) / 2 == pytest.approx(1.0)


def test_macro_f1_over_present_classes_and_selection_rule():
    y = np.array([0, 0, 3, 3])
    assert macro_f1(y, y) == 1.0
    assert macro_f1(y, np.array([0, 0, 0, 0])) == pytest.approx((2 / 3 + 0) / 2)
    assert is_better(0.6, 1.0, 0.5, 0.1)
    assert is_better(0.5, 0.05, 0.5, 0.1)
    assert not is_better(0.5, 0.2, 0.5, 0.1)


# ── Loaders on fake videos / images ───────────────────────────────────────────

def _write_video(path: Path, colors, fps=30.0, size=(64, 48)):
    cv2 = pytest.importorskip("cv2")
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert w.isOpened()
    for c in colors:
        w.write(np.full((size[1], size[0], 3), c, dtype=np.uint8))
    w.release()


def _daisee_root(tmp_path, clips):
    """clips: {clip_id: (split_dir, engagement, colors)}"""
    for split in ("Train", "Validation", "Test"):
        (tmp_path / "DataSet" / split).mkdir(parents=True, exist_ok=True)
    (tmp_path / "Labels").mkdir(exist_ok=True)
    rows: dict[str, list] = {"Train": [], "Validation": [], "Test": []}
    for cid, (split, eng, colors) in clips.items():
        d = tmp_path / "DataSet" / split / cid[:6] / cid
        d.mkdir(parents=True)
        _write_video(d / f"{cid}.avi", colors)
        rows[split].append(cid)
    for split, fname in (("Train", "TrainLabels.csv"), ("Validation", "ValidationLabels.csv"),
                         ("Test", "TestLabels.csv")):
        with open(tmp_path / "Labels" / fname, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["ClipID", "Boredom", "Engagement", "Confusion", "Frustration "])
            for cid in rows[split]:
                wr.writerow([f"{cid}.avi", 0, clips[cid][1], 0, 0])
    return tmp_path


def test_daisee_loader_on_fake_videos(tmp_path):
    pytest.importorskip("cv2")
    face = [(128, 200, 0)] * 120                       # 4 s @ 30 fps, face, level head
    noface = [(0, 0, 0)] * 120
    root = _daisee_root(tmp_path, {
        "1100011001": ("Train", 3, face),
        "1100011002": ("Train", 1, face),
        "1100021001": ("Train", 0, noface),
        "1100031001": ("Validation", 2, face),
    })
    from training.datasets.daisee_loader import DAiSEEDataset
    det = _ColorFaceDet()
    ds = DAiSEEDataset(str(root), split="train", face_detector=det, stride=5)
    seqs = list(ds.iter_sequences())
    # 4 s at 30 fps -> 5 Hz: 20 frames per clip, timestamps 200 ms apart
    assert [len(s) for s in seqs] == [20, 20, 20]
    assert np.allclose(np.diff(seqs[0].timestamps_ms), 200)
    assert det.calls == 60                              # landmarker ran at 5 Hz, not 30
    samples = list(DAiSEEDataset(str(root), split="train", face_detector=_ColorFaceDet(), stride=5))
    groups = {g for _, _, g in samples}
    assert groups == {"daisee:110001"}                  # no-face clip fully dropped, same subject grouped
    labels = {BEHAVIOR_CLASSES[y] for _, y, _ in samples}
    assert labels == {"attentive_screen", "distracted"}
    w = samples[0][0]
    assert w.shape == (WINDOW_SIZE, FEATURE_DIM)
    assert not np.allclose(w, 0)
    ds2 = DAiSEEDataset(str(root), split="train", face_detector=_ColorFaceDet(), stride=5)
    list(ds2)
    assert ds2.dropped_absent_windows == 2              # (20 - 15) / 5 + 1 windows of the no-face clip
    val = list(DAiSEEDataset(str(root), split="val", face_detector=_ColorFaceDet(), stride=5))
    assert {g for _, _, g in val} == {"daisee:110003"}


def test_engagenet_loader_on_fake_images(tmp_path):
    cv2 = pytest.importorskip("cv2")
    img_dir = tmp_path / "images"
    rows = []
    for subj, split, score in (("S1", "train", 0.9), ("S2", "train", 0.1), ("S3", "val", 0.4)):
        d = img_dir / subj / "c1"
        d.mkdir(parents=True)
        for i in range(40):                                 # 40 frames at 10 fps = 4 s
            cv2.imwrite(str(d / f"{i:04d}.png"), np.full((48, 64, 3), (128, 200, 0), dtype=np.uint8))
            rows.append({"image": f"{subj}/c1/{i:04d}.png", "score": score, "split": split, "frame": i})
    with open(tmp_path / "annotations.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["image", "score", "split", "frame"])
        wr.writeheader()
        wr.writerows(rows)
    from training.datasets.engagenet_loader import EngageNetDataset
    ds = EngageNetDataset(str(tmp_path), split="train", source_fps=10.0, face_detector=_ColorFaceDet(), stride=5)
    seqs = list(ds.iter_sequences())
    assert [len(s) for s in seqs] == [20, 20]
    assert [s.group_id for s in seqs] == ["engagenet:S1", "engagenet:S2"]
    samples = list(EngageNetDataset(str(tmp_path), split="train", source_fps=10.0,
                                    face_detector=_ColorFaceDet(), stride=5))
    assert {BEHAVIOR_CLASSES[y] for _, y, _ in samples} == {"attentive_screen", "drowsy"}
    val = list(EngageNetDataset(str(tmp_path), split="val", source_fps=10.0,
                                face_detector=_ColorFaceDet(), stride=5))
    assert {g for _, _, g in val} == {"engagenet:S3"}
    assert {BEHAVIOR_CLASSES[y] for _, y, _ in val} == {"distracted"}


def _classroom_root(tmp_path, n_frames=90, fps=15.0, split="test"):
    (tmp_path / "videos").mkdir(parents=True)
    (tmp_path / "annotations").mkdir()
    _write_video(tmp_path / "videos" / "clip1.avi", [(128, 200, 0)] * n_frames, fps=fps, size=(320, 240))
    persons = [{
        "track_id": 3, "seat_id": "seat_1", "subject_id": "stu9",
        "frames": [{"frame_idx": i, "bbox_xyxy": [10, 10, 150, 230],
                    "behavior": "attentive_reading" if i >= 45 else "attentive_screen",
                    "detected_objects": ["book"] if i >= 45 else [],
                    "hand_proximity": "holding_book" if i >= 45 else "hands_not_visible"}
                   for i in range(n_frames)],
    }]
    (tmp_path / "annotations" / "clip1.json").write_text(json.dumps(
        {"clip_id": "clip1", "fps": fps, "persons": persons, "n_frames": n_frames}))
    with open(tmp_path / "metadata.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip_id", "split", "duration_sec", "num_persons"])
        wr.writerow(["clip1", split, n_frames / fps, 1])
    return tmp_path


def test_classroom_loader_on_fake_video(tmp_path):
    pytest.importorskip("cv2")
    root = _classroom_root(tmp_path)
    from training.datasets.classroom_loader import ClassroomDataset
    ds = ClassroomDataset(str(root), split="test", face_detector=_ColorFaceDet(), stride=1)
    seqs = list(ds.iter_sequences())
    assert len(seqs) == 1
    s = seqs[0]
    assert len(s) == 30                                  # 6 s at 15 fps -> 5 Hz
    assert s.group_id == "classroom:stu9" and s.stream_id.startswith("classroom:clip1:3")
    assert s.vectors[-1][7] == 1.0 and s.vectors[0][7] == 0.0     # book channel from annotations
    samples = list(ClassroomDataset(str(root), split="test", face_detector=_ColorFaceDet(), stride=1))
    assert len(samples) == 30 - WINDOW_SIZE + 1
    assert list(ClassroomDataset(str(root), split="train", face_detector=_ColorFaceDet())) == []


def test_segment_sequence_splits_at_gaps():
    enc = TrackSequenceEncoder()
    s = LabelledSequence("x", "g")
    for t in [0, 200, 400, 2000, 2200]:
        s.frames.append(enc.step(t, _obs()))
        s.labels.append(0)
    parts = segment_sequence(s)
    assert [len(p) for p in parts] == [3, 2] and parts[0].stream_id == "x#0"


def test_encode_frame_v2_is_the_shared_encoder():
    """Guard: the encoder uses encode_frame_v2 (same as BehaviorClassifier._extract)."""
    enc = TrackSequenceEncoder()
    f = enc.step(0, _obs(pitch=10))
    assert np.array_equal(f.vector, encode_frame_v2(f.metrics, [], "hands_not_visible", None))
