#!/usr/bin/env python3
"""Verify that all expected sensor values are received over UART.

Example:
  python hardware_mock/verify_sensor_stream.py --port /dev/ttyUSB0 --seconds 20
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import serial


SENSOR_FIELDS = {
    "dht22": ["temp", "hum", "dht"],
    "mq135": ["air"],
    "ldr": ["light"],
    "hcsr04": ["dist"],
    "mpu6050": ["accel", "gx", "gy", "gz", "mpu"],
    "acs712": ["cur"],
}


@dataclass
class Stats:
    packets: int = 0
    json_errors: int = 0
    sensor_hits: dict[str, int] = field(default_factory=dict)
    missing_fields: dict[str, set[str]] = field(default_factory=dict)


def init_stats() -> Stats:
    stats = Stats()
    stats.sensor_hits = {name: 0 for name in SENSOR_FIELDS}
    stats.missing_fields = {name: set(fields) for name, fields in SENSOR_FIELDS.items()}
    return stats


def update_stats(stats: Stats, payload: dict) -> None:
    stats.packets += 1
    for sensor, fields in SENSOR_FIELDS.items():
        if all(field in payload for field in fields):
            stats.sensor_hits[sensor] += 1
            for field in fields:
                stats.missing_fields[sensor].discard(field)


def print_report(stats: Stats, duration_s: float) -> None:
    print("\n=== UART Sensor Verification Report ===")
    print(f"Duration: {duration_s:.1f}s")
    print(f"JSON packets: {stats.packets}")
    print(f"JSON errors : {stats.json_errors}")
    print()

    all_ok = True
    for sensor, fields in SENSOR_FIELDS.items():
        hits = stats.sensor_hits[sensor]
        missing = sorted(stats.missing_fields[sensor])
        if hits > 0 and not missing:
            print(f"[PASS] {sensor:8}  packets_with_all_fields={hits}")
        else:
            all_ok = False
            print(f"[FAIL] {sensor:8}  packets_with_all_fields={hits}  missing={missing}")

    print()
    if all_ok:
        print("Result: ALL EXPECTED SENSORS ARE PRESENT IN THE STREAM")
    else:
        print("Result: SOME SENSOR FIELDS ARE MISSING OR NEVER RECEIVED")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify UART JSON stream contains all sensor fields")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--seconds", type=float, default=20.0, help="Capture duration in seconds")
    args = parser.parse_args()

    stats = init_stats()
    start = time.monotonic()
    end = start + args.seconds
    buffer = ""

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.25)
    except serial.SerialException as exc:
        print(f"ERROR: cannot open {args.port}: {exc}")
        return 1

    print(f"Listening on {args.port} @ {args.baud} for {args.seconds:.1f}s...")

    try:
        while time.monotonic() < end:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue

            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    stats.json_errors += 1
                    continue

                update_stats(stats, payload)
    finally:
        ser.close()

    print_report(stats, time.monotonic() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())