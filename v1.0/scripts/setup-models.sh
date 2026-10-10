#!/usr/bin/env bash
# setup-models.sh — Prepare model and WASM assets for v1.0
# Run from the v1.0/ directory:  bash scripts/setup-models.sh
#
# 1. copies the onnxruntime-web WASM binaries to public/onnx/ (legacy detector)
# 2. downloads the pinned MediaPipe models used by DACE (Apache-2.0) to public/models/
# 3. copies the @mediapipe/tasks-vision WASM to public/mediapipe/wasm/
# 4. verifies every file listed in scripts/models.sha256 (exit 1 on mismatch)
# 5. prints instructions for the YOLOv8n model (AGPL-3.0; decision R2), which is
#    NOT downloaded automatically
#
# Model binaries and the copied WASM are gitignored; never commit them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKSUMS="$SCRIPT_DIR/models.sha256"

MP_BASE="https://storage.googleapis.com/mediapipe-models"
# Pinned model URLs (versioned paths). Changing one requires updating models.sha256.
FACE_URL="$MP_BASE/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
HAND_URL="$MP_BASE/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
EFFDET_URL="$MP_BASE/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
    else shasum -a 256 "$1" | awk '{print $1}'; fi
}

expected_sha() { # $1 = path relative to v1.0
    awk -v p="$1" '$0 !~ /^#/ && NF == 2 && $2 == p {print $1}' "$CHECKSUMS"
}

echo "[setup-models] Copying onnxruntime-web WASM binaries..."
ORT_WASM_SRC="$ROOT/node_modules/onnxruntime-web/dist"
ORT_WASM_DST="$ROOT/public/onnx"
mkdir -p "$ORT_WASM_DST"

if [ -d "$ORT_WASM_SRC" ]; then
    cp "$ORT_WASM_SRC"/*.wasm "$ORT_WASM_DST/" 2>/dev/null || true
    cp "$ORT_WASM_SRC"/*.mjs  "$ORT_WASM_DST/" 2>/dev/null || true
    echo "[setup-models] WASM binaries copied to public/onnx/"
else
    echo "[setup-models] WARNING: onnxruntime-web not found in node_modules."
    echo "  Run: npm install  (from v1.0/)"
fi

echo ""
echo "[setup-models] MediaPipe models (DACE; Apache-2.0)"
mkdir -p "$ROOT/public/models"

download() { # $1 = url, $2 = path relative to v1.0
    local url="$1" rel="$2" dst="$ROOT/$2" want have tmp
    want="$(expected_sha "$rel")"
    if [ -f "$dst" ] && [ -n "$want" ] && [ "$(sha256_of "$dst")" = "$want" ]; then
        echo "  ok (present)  $rel"
        return 0
    fi
    tmp="$(mktemp "${TMPDIR:-/tmp}/cognen-model.XXXXXX")"
    echo "  downloading   $rel"
    if ! curl -fL --retry 3 --connect-timeout 20 -o "$tmp" "$url"; then
        rm -f "$tmp"
        echo "  ERROR: download failed: $url" >&2
        return 1
    fi
    have="$(sha256_of "$tmp")"
    if [ -n "$want" ] && [ "$have" != "$want" ]; then
        rm -f "$tmp"
        echo "  ERROR: checksum mismatch for $rel (expected $want, got $have)" >&2
        return 1
    fi
    if [ -z "$want" ]; then
        echo "  WARNING: no checksum recorded for $rel; got $have"
        echo "           record it in scripts/models.sha256 before relying on this file."
    fi
    mv "$tmp" "$dst"
}

download "$FACE_URL"   "public/models/face_landmarker.task"
download "$HAND_URL"   "public/models/hand_landmarker.task"
download "$EFFDET_URL" "public/models/efficientdet_lite0.tflite"

echo ""
echo "[setup-models] Copying @mediapipe/tasks-vision WASM to public/mediapipe/wasm/"
MP_WASM_SRC="$ROOT/node_modules/@mediapipe/tasks-vision/wasm"
MP_WASM_DST="$ROOT/public/mediapipe/wasm"
if [ -d "$MP_WASM_SRC" ]; then
    mkdir -p "$MP_WASM_DST"
    cp "$MP_WASM_SRC"/* "$MP_WASM_DST/"
else
    echo "  ERROR: @mediapipe/tasks-vision not installed. Run: npm install" >&2
    exit 1
fi

echo ""
echo "[setup-models] Verifying scripts/models.sha256"
fail=0
while read -r sum rel; do
    case "$sum" in ''|\#*) continue ;; esac
    f="$ROOT/$rel"
    if [ ! -f "$f" ]; then
        echo "  MISSING  $rel"; fail=1; continue
    fi
    if [ "$(sha256_of "$f")" = "$sum" ]; then echo "  OK       $rel"; else echo "  MISMATCH $rel"; fail=1; fi
done < "$CHECKSUMS"
if [ "$fail" -ne 0 ]; then
    echo "[setup-models] Checksum verification FAILED." >&2
    echo "  If @mediapipe/tasks-vision was upgraded on purpose, re-record the WASM hashes in scripts/models.sha256." >&2
    exit 1
fi

echo ""
echo "[setup-models] YOLOv8-nano object detection model (optional; AGPL-3.0)"
MODEL_DST="$ROOT/public/models/yolov8n-objects.onnx"
if [ -f "$MODEL_DST" ]; then
    echo "[setup-models] Model present: public/models/yolov8n-objects.onnx (sha256 $(sha256_of "$MODEL_DST"))"
    echo "  To make the browser verify it: record this hash in scripts/models.sha256 and set"
    echo "  NEXT_PUBLIC_DACE_YOLO_SHA256 to the same value."
else
    echo "  Not downloaded automatically. YOLOv8 (Ultralytics) is AGPL-3.0: under decision R2 it may be"
    echo "  used for RESEARCH only until the supervisors approve it (or a commercial licence is confirmed)."
    echo "  The Apache-2.0 alternative is the MediaPipe EfficientDet-Lite0 detector downloaded above"
    echo "  (DACE: NEXT_PUBLIC_DACE_DETECTOR=mediapipe)."
    echo "  To export it for research use (Python + ultralytics):"
    echo "       pip install ultralytics"
    echo "       python3 -c \""
    echo "         from ultralytics import YOLO"
    echo "         m = YOLO('yolov8n.pt')"
    echo "         m.export(format='onnx', imgsz=640, simplify=True)"
    echo "         import shutil"
    echo "         shutil.move('yolov8n.onnx', 'public/models/yolov8n-objects.onnx')"
    echo "       \""
    echo "  Then re-run this script to print its hash. Without the model, DACE runs with"
    echo "  detectorAvailable=false (hand-only evidence; FR4 self-test gate)."
fi

echo ""
echo "[setup-models] Done."
