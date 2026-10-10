"""
Training loop + ONNX export: leakage-free split recorded in the checkpoint,
macro-F1 model selection, seeded determinism, ONNX/PyTorch parity check.
Requires torch + onnxruntime (heavy tier); skipped otherwise.

Run: pytest v2.0/tests/test_training_export.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

V2_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V2_ROOT))

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
pytest.importorskip("sklearn")


def _args(tmp_path, name, **kw):
    a = dict(daisee=None, engagenet=None, classroom=None, output=str(tmp_path / f"{name}.pt"),
             epochs=2, batch=64, lr=1e-3, patience=5, val_ratio=0.15, max_clips_per_dataset=None,
             realistic=True, realistic_n_per_class=30, realistic_augment=1, realistic_seed=3,
             seed=11, penalty_loss=True)
    a.update(kw)
    return argparse.Namespace(**a)


def test_train_records_split_and_macro_f1_and_is_deterministic(tmp_path):
    from training.train_classifier import train
    r1 = train(_args(tmp_path, "a"))
    r2 = train(_args(tmp_path, "b"))
    ck = torch.load(r1["checkpoint"], weights_only=True)
    assert ck["selection_metric"].startswith("val_macro_f1")
    assert 0.0 <= ck["val_macro_f1"] <= 1.0
    sp = ck["split"]
    # synthetic loader split kept: n_val = int(30 * 0.15) = 4 base windows per class,
    # 26 train base windows per class + 1 augmented copy each
    assert sp["n_train"] == 6 * 26 * 2 and sp["n_val"] == 6 * 4
    assert sp["val_windows_removed_for_leakage"] == 0
    ck2 = torch.load(r2["checkpoint"], weights_only=True)
    for k, v in ck["model_state_dict"].items():
        assert torch.equal(v, ck2["model_state_dict"][k]), k
    assert Path(r1["onnx"]).exists()


def test_onnx_parity_check_passes_and_catches_a_mismatch(tmp_path):
    from training.export_onnx import OnnxParityError, check_onnx_parity, export_onnx
    from training.model import BehaviorClassifierCNN
    m = BehaviorClassifierCNN()
    # non-trivial BN statistics so eval-mode export matters
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.uniform_(-0.5, 0.5)
                mod.running_var.uniform_(0.5, 2.0)
    ck = tmp_path / "m.pt"
    torch.save({"model_state_dict": m.state_dict()}, ck)
    onnx_path = export_onnx(str(ck), str(tmp_path / "m.onnx"))
    m.eval()
    assert check_onnx_parity(m, onnx_path) < 1e-4
    other = BehaviorClassifierCNN()            # different weights -> must fail
    with pytest.raises(OnnxParityError):
        check_onnx_parity(other, onnx_path)


def test_shipped_v3_onnx_matches_its_checkpoint():
    ckpt = V2_ROOT / "models" / "behavior_classifier_v3.pt"
    onnx_path = V2_ROOT / "models" / "behavior_classifier_v3.onnx"
    if not (ckpt.exists() and onnx_path.exists()):
        pytest.skip("v3 model files not present")
    from training.export_onnx import check_onnx_parity
    from training.model import BehaviorClassifierCNN
    assert check_onnx_parity(BehaviorClassifierCNN.load(str(ckpt)), onnx_path) < 1e-4


def test_iter_batches_is_seeded():
    from training.train_classifier import _iter_batches
    X = np.arange(20)[:, None, None].astype(np.float32)
    y = np.arange(20)
    a = [b[1].tolist() for b in _iter_batches(X, y, 7, rng=np.random.default_rng(1))]
    b = [b[1].tolist() for b in _iter_batches(X, y, 7, rng=np.random.default_rng(1))]
    assert a == b and sorted(sum(a, [])) == list(range(20))
