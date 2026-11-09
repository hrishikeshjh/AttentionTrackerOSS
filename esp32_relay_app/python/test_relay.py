#!/usr/bin/env python3
"""
CognEn Quick Relay Hardware Diagnostic Tool
Tests direct serial communication with ESP32 and cycles all 4 relays.
"""

import sys
import time
import glob
import serial

def main():
    ports = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    if not ports:
        print("[ERROR] No ESP32 serial port found (/dev/ttyACM* or /dev/ttyUSB*).")
        sys.exit(1)

    port = ports[0]
    print(f"[*] Detected port: {port}")
    try:
        ser = serial.Serial(port, 115200, timeout=2)
    except PermissionError:
        print(f"[ERROR] Permission denied on {port}!")
        print(f"Run: sudo chmod 666 {port}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not open {port}: {e}")
        sys.exit(1)

    time.sleep(2.0) # Wait for ESP32 boot
    ser.reset_input_buffer()

    print("[*] Sending PING...")
    ser.write(b"PING\n")
    resp = ser.readline().decode().strip()
    print(f"[*] Response: '{resp}'")

    print("\n[*] Starting Relay Cycle Test:")
    for ch in range(1, 5):
        print(f" -> Turning Relay CH {ch} ON...")
        ser.write(f"CH{ch}:1\n".encode())
        time.sleep(0.3)
        ack = ser.readline().decode().strip()
        print(f"    ESP32 Ack: {ack}")
        time.sleep(1.0)

        print(f" -> Turning Relay CH {ch} OFF...")
        ser.write(f"CH{ch}:0\n".encode())
        time.sleep(0.3)
        ack = ser.readline().decode().strip()
        print(f"    ESP32 Ack: {ack}")
        time.sleep(0.5)

    print("\n[*] Cycling ALL Relays (ALL:1111)...")
    ser.write(b"ALL:1111\n")
    time.sleep(0.3)
    print("    Ack:", ser.readline().decode().strip())
    time.sleep(1.5)

    print("[*] Turning ALL Relays OFF (ALL:0000)...")
    ser.write(b"ALL:0000\n")
    time.sleep(0.3)
    print("    Ack:", ser.readline().decode().strip())

    ser.close()
    print("\n[+] Diagnostic test completed.")

if __name__ == "__main__":
    main()
