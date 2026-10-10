"""
export_onnx.py — Export a trained BehaviorClassifierCNN to ONNX.

Usage:
    python3 v2.0/training/export_onnx.py \\
        --checkpoint v2.0/models/behavior_classifier.pt \\
        --output     v2.0/models/behavior_classifier.onnx

The exported ONNX model:
    - Input  : "input"  — float32 (batch, F, WINDOW_SIZE), F = the checkpoint's
               channel count: 15 for FEATURE_VECTOR_VERSION 2 models
               (behavior_classifier_v3), 13 for older ones.
    - Output : "logits" — float32 (batch, NUM_CLASSES) = (B, 6)
    - Dynamic batch dimension for use in onnxruntime (Python) and
      onnxruntime-web (browser / v1.0 TemporalBehaviorClassifier.loadModel:
      input 'input', channels-first (1, F, 15), output 'logits' + softmax,
      v2.0 proximity codes; pass ``featureDim: 15`` for a v3 model, which
      makes v1.0 append the unknown-posture channels (1, 0)).
      Contract pinned by tests/test_cross_language_contract.py and
      tests/test_feature_vector_v2_contract.py.
    - Sign convention of the inputs: pitch_norm / gaze_y are DOWN = NEGATIVE
      (features/feature_vector.py). v2.0 measures this with the MediaPipe
      transformation matrix + blendshapes (features/dace_measurement.py);
      v1.0 DACE mode feeds the same convention (negated DACE values, plan
      D7). v1.0's legacy (DACE-off) path does not. No model has been trained
      on real data for either surface.

ONNX opset: 17 (supported by onnxruntime ≥ 1.15 and onnxruntime-web ≥ 1.16).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("cognen.export_onnx")


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu",
    opset: int = 17,
    verify_parity: bool = True,
) -> Path:
    """
    Load a PyTorch checkpoint and export to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        output_path:     Destination .onnx path.
        device:          "cpu" or "cuda".
        opset:           ONNX opset version.
        verify_parity:   compare ONNX vs PyTorch logits (check_onnx_parity);
                         raises OnnxParityError on a mismatch.

    Returns:
        Path to the written ONNX file.
    """
    import torch

    from training.model import FEATURE_DIM, WINDOW_SIZE, BehaviorClassifierCNN

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = BehaviorClassifierCNN.load(checkpoint_path, device=device)
    model.eval()

    # Dummy input: batch=1, the checkpoint's channel count (FEATURE_DIM=15 for
    # v3 checkpoints), window=15
    if model.feature_dim != FEATURE_DIM:
        logger.warning("checkpoint has %d input channels (current FEATURE_DIM=%d)", model.feature_dim, FEATURE_DIM)
    dummy = torch.zeros(1, model.feature_dim, WINDOW_SIZE, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        # torch >= 2.9 defaults to the dynamo exporter (needs onnxscript,
        # emits opset 18 + an external .onnx.data sidecar). The TorchScript
        # exporter keeps the single-file opset-17 contract v1.0 loadModel expects.
        dynamo=False,
    )

    logger.info("Exported ONNX model → %s", output_path)

    # Structural check: a failure is an error now (it used to be logged and
    # the broken file kept).
    try:
        import onnx
    except ImportError:
        logger.warning("onnx package not installed; skipping the structural model check.")
    else:
        onnx.checker.check_model(onnx.load(str(output_path)))
        logger.info("ONNX model check passed.")

    # Numerical parity: the ONNX graph must reproduce the PyTorch logits.
    if verify_parity:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            logger.warning("onnxruntime not installed; skipping the ONNX/PyTorch parity check.")
        else:
            diff = check_onnx_parity(model, output_path, device=device)
            logger.info("ONNX/PyTorch parity: max |Δlogit| = %.2e", diff)

    return output_path


class OnnxParityError(AssertionError):
    """The exported ONNX model's logits differ from the PyTorch model's."""


def check_onnx_parity(
    model,
    onnx_path: str | Path,
    n: int = 64,
    atol: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    """
    Run *n* random windows (feature-range inputs: uniform [-1, 1]) through the
    PyTorch model (eval mode) and the ONNX model (onnxruntime, CPU); raise
    OnnxParityError when the largest absolute logit difference exceeds
    *atol*. Also checks batch-size dynamics (batch 1 and batch n). Returns
    the max abs difference.
    """
    import numpy as np
    import onnxruntime as ort
    import torch

    from training.model import WINDOW_SIZE

    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, model.feature_dim, WINDOW_SIZE)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        ref = model(torch.from_numpy(x).to(device)).cpu().numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input": x})[0]
    got1 = sess.run(["logits"], {"input": x[:1]})[0]
    if got.shape != ref.shape or got1.shape != (1, ref.shape[1]):
        raise OnnxParityError(f"shape mismatch: onnx {got.shape}/{got1.shape} vs torch {ref.shape}")
    diff = float(max(np.max(np.abs(got - ref)), np.max(np.abs(got1 - ref[:1]))))
    if not np.isfinite(diff) or diff > atol:
        raise OnnxParityError(f"ONNX logits differ from PyTorch by {diff:.3e} (atol {atol:.0e})")
    return diff


def verify_onnx_runtime(onnx_path: str, device: str = "cpu") -> None:
    """
    Run a quick inference check with onnxruntime to verify the exported model.
    """
    try:
        import numpy as np
        import onnxruntime as ort

        from training.model import WINDOW_SIZE

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" \
                    else ["CPUExecutionProvider"]
        sess  = ort.InferenceSession(onnx_path, providers=providers)
        feature_dim = int(sess.get_inputs()[0].shape[1])
        dummy = np.zeros((1, feature_dim, WINDOW_SIZE), dtype=np.float32)
        out   = sess.run(["logits"], {"input": dummy})
        logits = out[0]
        if logits.shape != (1, 6):
            raise OnnxParityError(f"Unexpected output shape: {logits.shape}")
        logger.info("ORT verification passed: logits shape %s", logits.shape)
    except ImportError:
        logger.warning("onnxruntime not installed; skipping ORT verification.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export BehaviorClassifierCNN to ONNX")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--output",     type=str,
                   default=str(Path(__file__).parent.parent / "models" / "behavior_classifier.onnx"),
                   help="Output .onnx path")
    p.add_argument("--device",  type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--opset",   type=int, default=17)
    p.add_argument("--verify",  action="store_true", help="Run ORT verification after export")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    args = _parse_args()
    onnx_path = export_onnx(args.checkpoint, args.output, args.device, args.opset)
    if args.verify:
        verify_onnx_runtime(str(onnx_path), args.device)
