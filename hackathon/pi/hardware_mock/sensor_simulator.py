"""
Hardware mock – simulates all 8 sensors and POSTs to FastAPI /ingest at 1 Hz.
Run: python -m apps.pi.hardware_mock.sensor_simulator
  or via Makefile: make mock
"""
from __future__ import annotations
import math
import random
import time
import argparse
import urllib.request
import urllib.error
import json

API_URL = "http://localhost:8000/ingest"


def _sin_wave(t: float, period: float, offset: float, amplitude: float, base: float) -> float:
    return base + amplitude * math.sin(2 * math.pi * t / period + offset)


def generate_reading(t: float, scenario: str = "normal") -> dict:
    """Generate a realistic sensor snapshot at time t (seconds since start)."""

    ts = int(time.time())

    # ── DHT22 ────────────────────────────────────────────────────────────────
    air_temp = _sin_wave(t, 120, 0, 0.8, 34.5)
    humidity = _sin_wave(t, 180, 1, 3.0, 55.0)

    # ── BMP280 ────────────────────────────────────────────────────────────────
    pressure = _sin_wave(t, 600, 2, 2.0, 1013.0)
    altitude = _sin_wave(t, 600, 2, 0.5, 95.0)
    bmp_temp = air_temp + random.gauss(0, 0.3)

    # ── MQ-135 ───────────────────────────────────────────────────────────────
    aq_raw = int(_sin_wave(t, 300, 3, 150, 600) + random.gauss(0, 20))
    aq_ppm = aq_raw * 500.0 / 4095.0

    # ── LDR ──────────────────────────────────────────────────────────────────
    light_raw = int(_sin_wave(t, 86400, 0, 1000, 1500) + random.gauss(0, 50))
    light_raw = max(0, min(4095, light_raw))
    light_lux = 500000.0 / max(1, (3.3 - light_raw * 3.3 / 4095) / max(0.001, light_raw * 3.3 / 4095) * 10000)

    # ── HC-SR04 ──────────────────────────────────────────────────────────────
    lid_distance = 2.0 + random.gauss(0, 0.2)  # closed
    lid_open = lid_distance > 5.0

    # ── MPU6050 ──────────────────────────────────────────────────────────────
    accel_x = random.gauss(0, 0.03)
    accel_y = random.gauss(0, 0.03)
    accel_z = 1.0 + random.gauss(0, 0.03)
    gyro_x  = random.gauss(0, 0.5)
    gyro_y  = random.gauss(0, 0.5)
    gyro_z  = random.gauss(0, 0.5)
    mpu_temp = 35.0 + random.gauss(0, 0.5)

    # ── ACS712 ───────────────────────────────────────────────────────────────
    heater_current = 0.52 + random.gauss(0, 0.02)
    heater_active = heater_current > 0.1

    # ── SG90 servo ───────────────────────────────────────────────────────────
    servo_angle = 0

    # ── Apply scenario overrides ─────────────────────────────────────────────
    if scenario == "hyperthermia":
        air_temp = 40.0 + random.gauss(0, 0.3)
    elif scenario == "heaterFail":
        heater_current = 0.0
        heater_active = False
    elif scenario == "lidOpen":
        lid_distance = 18.0 + random.gauss(0, 0.5)
        lid_open = True
    elif scenario == "poorAir":
        aq_raw = 3000
        aq_ppm = aq_raw * 500.0 / 4095.0
    elif scenario == "vibration":
        accel_x = random.gauss(0, 0.8)
        accel_y = random.gauss(0, 0.8)
    elif scenario == "ventActive":
        servo_angle = 90

    return {
        "ts": ts,
        "airTempC": round(air_temp, 2),
        "humidityPct": round(humidity, 2),
        "pressureHpa": round(pressure, 2),
        "altitudeM": round(altitude, 2),
        "bmpTempC": round(bmp_temp, 2),
        "airQualityRaw": aq_raw,
        "airQualityPpm": round(aq_ppm, 1),
        "lightRaw": light_raw,
        "lightLux": round(light_lux, 1),
        "lidDistanceCm": round(lid_distance, 1),
        "lidOpen": lid_open,
        "accelX": round(accel_x, 4),
        "accelY": round(accel_y, 4),
        "accelZ": round(accel_z, 4),
        "gyroX": round(gyro_x, 2),
        "gyroY": round(gyro_y, 2),
        "gyroZ": round(gyro_z, 2),
        "mpuTempC": round(mpu_temp, 2),
        "heaterCurrentA": round(heater_current, 4),
        "heaterActive": heater_active,
        "servoAngleDeg": servo_angle,
    }


def post(payload: dict) -> int:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status
    except urllib.error.URLError as e:
        print(f"  [mock] POST failed: {e}")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="NeoTwin sensor simulator")
    parser.add_argument("--scenario", default="normal",
        choices=["normal", "hyperthermia", "heaterFail", "lidOpen", "poorAir", "vibration", "ventActive"],
        help="Scenario to simulate")
    parser.add_argument("--hz", type=float, default=1.0, help="Posts per second (default 1)")
    args = parser.parse_args()

    interval = 1.0 / args.hz
    print(f"[NeoTwin Mock] scenario={args.scenario}  rate={args.hz}Hz -> {API_URL}")
    print("Press Ctrl+C to stop.\n")

    t = 0.0
    while True:
        reading = generate_reading(t, args.scenario)
        status = post(reading)
        print(
            f"  t={t:6.0f}s | T={reading['airTempC']}C "
            f"H={reading['humidityPct']}% "
            f"P={reading['pressureHpa']}hPa "
            f"d={reading['lidDistanceCm']}cm "
            f"I={reading['heaterCurrentA']}A "
            f"AQ={reading['airQualityRaw']} "
            f"lux={reading['lightLux']} "
            f"-> HTTP {status}"
        )
        t += interval
        time.sleep(interval)


if __name__ == "__main__":
    main()
