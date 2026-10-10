"""
export_onnx.py — Export a trained BehaviorClassifierCNN to ONNX.

Usage:
    python3 v2.0/training/export_onnx.py \\
        --checkpoint v2.0/models/behavior_classifier.pt \\
        --output     v2.0/models/behavior_classifier.onnx

The exported ONNX model:
    - Input  : "input"  — float32 (batch, FEATURE_DIM, WINDOW_SIZE) = (B, 13, 15)
    - Output : "logits" — float32 (batch, NUM_CLASSES) = (B, 6)
    - Dynamic batch dimension for use in onnxruntime (Python) and
      onnxruntime-web (browser / v1.0 TemporalBehaviorClassifier.loadModel,
      whose defaults match this contract: input 'input', channels-first
      (1, 13, 15), output 'logits' + softmax, v2.0 proximity codes).
      Contract pinned by tests/test_cross_language_contract.py. Caveat: in
      v1.0 DACE mode the pitch/gaze_y fed to the model are negated DACE
      values (degrees, down = negative; ARCHITECTURE.md §11.8), not this repo's
      68-point legacy formula, so the input distributions differ across the
      two surfaces; no model has been trained on real data for either.

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
) -> Path:
    """
    Load a PyTorch checkpoint and export to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        output_path:     Destination .onnx path.
        device:          "cpu" or "cuda".
        opset:           ONNX opset version.

    Returns:
        Path to the written ONNX file.
    """
    import torch

    from training.model import FEATURE_DIM, WINDOW_SIZE, BehaviorClassifierCNN

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = BehaviorClassifierCNN.load(checkpoint_path, device=device)
    model.eval()

    # Dummy input: batch=1, feature_dim=13, window=15
    dummy = torch.zeros(1, FEATURE_DIM, WINDOW_SIZE, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    logger.info("Exported ONNX model → %s", output_path)

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model check passed.")
    except ImportError:
        logger.warning("onnx package not installed; skipping model verification.")
    except Exception as exc:
        logger.error("ONNX model check failed: %s", exc)

    return output_path


def verify_onnx_runtime(onnx_path: str, device: str = "cpu") -> None:
    """
    Run a quick inference check with onnxruntime to verify the exported model.
    """
    try:
        import numpy as np
        import onnxruntime as ort

        from training.model import FEATURE_DIM, WINDOW_SIZE

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" \
                    else ["CPUExecutionProvider"]
        sess  = ort.InferenceSession(onnx_path, providers=providers)
        dummy = np.zeros((1, FEATURE_DIM, WINDOW_SIZE), dtype=np.float32)
        out   = sess.run(["logits"], {"input": dummy})
        logits = out[0]
        assert logits.shape == (1, 6), f"Unexpected output shape: {logits.shape}"
        logger.info("ORT verification passed: logits shape %s", logits.shape)
    except ImportError:
        logger.warning("onnxruntime not installed; skipping ORT verification.")
    except Exception as exc:
        logger.error("ORT verification failed: %s", exc)


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
