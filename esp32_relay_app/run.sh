#!/usr/bin/env bash
#
# CognEn ESP32 Relay Hub Launcher
# Includes Native Hardware Serial Bridge for zero-sandbox restriction
#

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PORT=8080

echo "=========================================================="
echo "   CognEn — Cyber-Physical 4-Channel Relay Control Hub   "
echo "   Project: TIH/RP/20 | NM-ICPS Mandate                   "
echo "=========================================================="
echo ""
echo "[*] Launching Native Hardware Bridge on port $PORT..."
echo ""

exec python3 "$DIR/python/server.py" "$PORT"
