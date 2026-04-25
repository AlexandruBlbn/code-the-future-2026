#!/usr/bin/env python3
"""Simple UART link test for Raspberry Pi GPIO TX/RX/GND wiring.

Usage examples:
  python hardware_mock/uart_tester.py
  python hardware_mock/uart_tester.py --port /dev/ttyAMA10
  python hardware_mock/uart_tester.py --port auto --message HELLO

The script periodically sends a line over UART and prints any bytes received
back on the same connection. It is useful for checking:
  - physical TX/RX/GND wiring to an ESP32 or USB-UART adapter
  - loopback testing when TX is bridged to RX locally
  - whether the selected serial device is alive and readable
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import serial
from serial.tools import list_ports


def _resolve_port(requested_port: str) -> str:
    if requested_port and requested_port.lower() != "auto":
        return requested_port

    available_ports = [port.device for port in list_ports.comports()]
    preferred_ports = (
        "/dev/ttyAMA10",
        "/dev/ttyAMA0",
        "/dev/serial0",
        "/dev/ttyS0",
    )

    for candidate in preferred_ports:
        if candidate in available_ports:
            return candidate

    if available_ports:
        return available_ports[0]

    raise FileNotFoundError(
        "No serial ports were found. Connect the UART wires or pass --port explicitly."
    )


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="UART communication tester")
    parser.add_argument("--port", default="auto", help="Serial port (default: auto)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between pings")
    parser.add_argument("--message", default="PING", help="Message prefix to send")
    parser.add_argument("--count", type=int, default=0, help="Number of pings to send (0 = forever)")
    args = parser.parse_args()

    try:
        port = _resolve_port(args.port)
    except FileNotFoundError as exc:
        print(f"[{_stamp()}] ERROR {exc}")
        return 1

    print(f"[{_stamp()}] Opening {port} @ {args.baud} baud")

    try:
        ser = serial.Serial(port, args.baud, timeout=0.2, write_timeout=1.0)
    except serial.SerialException as exc:
        print(f"[{_stamp()}] ERROR Cannot open {port}: {exc}")
        return 1

    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print("[info] Type Ctrl+C to stop.")
    print("[info] If TX and RX are bridged, you should see your own sent lines echoed back.")

    sent = 0
    try:
        while True:
            sent += 1
            payload = f"{args.message} {sent} {_stamp()}"
            ser.write((payload + "\n").encode("utf-8"))
            ser.flush()
            print(f"[{_stamp()}] TX  {payload}")

            deadline = time.time() + max(0.5, args.interval)
            while time.time() < deadline:
                waiting = ser.in_waiting
                chunk = ser.read(waiting or 1)
                if chunk:
                    text = chunk.decode("utf-8", errors="replace")
                    for line in text.splitlines():
                        line = line.strip()
                        if line:
                            print(f"[{_stamp()}] RX  {line}")
                else:
                    time.sleep(0.05)

            if args.count and sent >= args.count:
                break

            time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        print(f"[{_stamp()}] Stopped by user.")
    finally:
        try:
            ser.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())