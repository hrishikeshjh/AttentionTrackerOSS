#!/usr/bin/env bash
# setup-models.sh — Prepare model and WASM assets for v1.0
# Run from the v1.0/ directory:  bash scripts/setup-models.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

echo "[setup-models] Copying onnxruntime-web WASM binaries..."
ORT_WASM_SRC="$ROOT/node_modules/onnxruntime-web/dist"
ORT_WASM_DST="$ROOT/public/onnx"

if [ -d "$ORT_WASM_SRC" ]; then
    cp "$ORT_WASM_SRC"/*.wasm "$ORT_WASM_DST/" 2>/dev/null || true
    cp "$ORT_WASM_SRC"/*.mjs  "$ORT_WASM_DST/" 2>/dev/null || true
    echo "[setup-models] WASM binaries copied to public/onnx/"
else
    echo "[setup-models] WARNING: onnxruntime-web not found in node_modules."
    echo "  Run: npm install  (from v1.0/)"
fi

echo ""
echo "[setup-models] YOLOv8-nano object detection model"
MODEL_DST="$ROOT/public/models/yolov8n-objects.onnx"

if [ -f "$MODEL_DST" ]; then
    echo "[setup-models] Model already present: public/models/yolov8n-objects.onnx"
else
    echo "[setup-models] Model not found. Options:"
    echo "  1. Download pre-exported ONNX (recommended for dev):"
    echo "       wget -O public/models/yolov8n-objects.onnx <URL>"
    echo "  2. Export from Ultralytics (requires Python + ultralytics):"
    echo "       pip install ultralytics"
    echo "       python3 -c \""
    echo "         from ultralytics import YOLO"
    echo "         m = YOLO('yolov8n.pt')"
    echo "         m.export(format='onnx', imgsz=640, simplify=True)"
    echo "         import shutil, pathlib"
    echo "         shutil.move('yolov8n.onnx', 'public/models/yolov8n-objects.onnx')"
    echo "       \""
fi

echo ""
echo "[setup-models] Done."
