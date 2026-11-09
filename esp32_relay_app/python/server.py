#!/usr/bin/env python3
"""
CognEn Unified Server & Hardware Serial Bridge
Serves the web dashboard on port 8080 and proxies relay commands directly to /dev/ttyACM0.
Bypasses all browser sandbox / snap confinement restrictions.
"""

import os
import sys
import json
import glob
import time
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, SimpleHTTPRequestHandler

try:
    import serial
except ImportError:
    print("[!] pyserial not found. Installing via uv/pip...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyserial"])
    import serial

# Detect Serial Port
def find_serial_port():
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not candidates:
        return None, None
    last_err = None
    for port in candidates:
        try:
            s = serial.Serial(port, 115200, timeout=1)
            s.close()
            return port, None
        except Exception as e:
            err_str = str(e)
            if "Permission denied" in err_str or (hasattr(e, "errno") and e.errno == 13):
                last_err = f"Permission denied on {port}. Please run: sudo chmod 666 {port}"
            else:
                last_err = f"Could not open {port}: {err_str}"
    return (candidates[0] if candidates else None), last_err

class SerialManager:
    def __init__(self):
        self.ser = None
        self.port = None
        self.last_error = None
        self.states = [0, 0, 0, 0]
        self.connect()

    def connect(self):
        port, err = find_serial_port()
        self.port = port
        self.last_error = err

        if err and "Permission denied" in err:
            print(f"[!] {err}")
            return False

        if not self.port:
            print("[-] No serial port detected yet. Standing by...")
            return False

        try:
            self.ser = serial.Serial(self.port, 115200, timeout=1)
            self.ser.dtr = False
            self.ser.rts = False
            time.sleep(2.0)  # Allow ESP32 initialization
            self.ser.write(b"PING\n")
            print(f"[+] Hardware Bridge connected to ESP32 on {self.port} (115200 baud).")
            self.last_error = None
            return True
        except Exception as e:
            print(f"[-] Could not open {self.port}: {e}")
            self.ser = None
            err_str = str(e)
            if "Permission denied" in err_str or (hasattr(e, "errno") and e.errno == 13):
                self.last_error = f"Permission denied on {self.port}. Please run: sudo chmod 666 {self.port}"
            else:
                self.last_error = f"Could not open {self.port}: {err_str}"
            return False

    def write(self, cmd: str):
        if not self.ser or not self.ser.is_open:
            if not self.connect():
                return False
        try:
            self.ser.write(cmd.encode())
            time.sleep(0.05)
            if self.ser.in_waiting:
                resp = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='replace')
                print(f"[ESP32 RX] {resp.strip()}")
            return True
        except Exception as e:
            print(f"[-] Write failed: {e}")
            self.ser = None
            return False

    def set_all(self, s: str):
        if len(s) == 4 and all(c in "01" for c in s):
            self.states = [int(c) for c in s]
            return self.write(f"ALL:{s}\n")
        return False

    def set_channel(self, ch: int, state: int):
        if 1 <= ch <= 4:
            self.states[ch - 1] = 1 if state else 0
            return self.write(f"CH{ch}:{1 if state else 0}\n")
        return False

    def pulse(self, ch: int, ms: int):
        if 1 <= ch <= 4:
            return self.write(f"PULSE:{ch}:{ms}\n")
        return False

serial_mgr = SerialManager()

class BridgeHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path.startswith("/api/"):
            params = parse_qs(parsed.query)

            if parsed.path == "/api/status":
                if not serial_mgr.ser or not serial_mgr.ser.is_open:
                    serial_mgr.connect()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                data = {
                    "connected": serial_mgr.ser is not None and serial_mgr.ser.is_open,
                    "port": serial_mgr.port or "None",
                    "error": serial_mgr.last_error,
                    "states": serial_mgr.states
                }
                self.wfile.write(json.dumps(data).encode())
                return

            elif parsed.path == "/api/all":
                s = params.get("s", ["0000"])[0]
                ok = serial_mgr.set_all(s)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": ok, "connected": serial_mgr.ser is not None and serial_mgr.ser.is_open, "states": serial_mgr.states}).encode())
                return

            elif parsed.path == "/api/relay":
                ch = int(params.get("ch", [1])[0])
                state = int(params.get("state", [0])[0])
                ok = serial_mgr.set_channel(ch, state)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": ok, "connected": serial_mgr.ser is not None and serial_mgr.ser.is_open, "channel": ch, "state": state}).encode())
                return

            elif parsed.path == "/api/pulse":
                ch = int(params.get("ch", [1])[0])
                ms = int(params.get("ms", [2000])[0])
                ok = serial_mgr.pulse(ch, ms)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": ok, "connected": serial_mgr.ser is not None and serial_mgr.ser.is_open, "channel": ch, "ms": ms}).encode())
                return

        # Otherwise serve static files from web/
        super().do_GET()

def run_server(port=8080):
    web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
    os.chdir(web_dir)
    server = HTTPServer(("0.0.0.0", port), BridgeHandler)
    print("==========================================================")
    print("   CognEn — Cyber-Physical 4-Channel Relay Control Hub   ")
    print("==========================================================")
    print(f"[*] Dashboard URL:       http://localhost:{port}")
    if serial_mgr.ser and serial_mgr.ser.is_open:
        print(f"[+] Native Serial Bridge: ACTIVE on {serial_mgr.port}")
    elif serial_mgr.last_error:
        print(f"[!] Hardware Warning:    {serial_mgr.last_error}")
    else:
        print("[-] Hardware Standby:    Connect ESP32 via USB anytime.")
    print("----------------------------------------------------------")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        if serial_mgr.ser:
            serial_mgr.ser.close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
