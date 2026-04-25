"""
AAP-based rules engine extended with all sensor inputs.
Returns severity level and list of active rule names.
"""
from __future__ import annotations
from .models import SensorReading, Severity

# ── Thresholds ────────────────────────────────────────────────────────────────
AIR_TEMP_MIN = 32.0
AIR_TEMP_MAX = 37.0
HUMIDITY_MIN = 50.0
HUMIDITY_MAX = 60.0
PRESSURE_LOW = 950.0   # hPa – decompression / altitude alarm
PRESSURE_HIGH = 1050.0
AIR_QUALITY_WARN = 1500.0  # ppm MQ-135 approx
AIR_QUALITY_CRIT = 2500.0
LIGHT_LUX_HIGH = 1000.0    # avoid intense light on neonate
LID_OPEN_CM = 5.0
HEATER_MIN_A = 0.1
VIBRATION_G_WARN = 0.4     # incubator vibration threshold
VIBRATION_G_CRIT = 1.0


def _max_accel(r: SensorReading) -> float:
    ax = r.accelX or 0.0
    ay = r.accelY or 0.0
    az = (r.accelZ or 1.0) - 1.0  # subtract gravity
    return max(abs(ax), abs(ay), abs(az))


def classify(reading: SensorReading) -> tuple[Severity, list[str]]:
    rules: list[str] = []
    worst: int = 0  # 0=normal 1=watch 2=alert 3=critical

    def flag(name: str, level: int) -> None:
        nonlocal worst
        rules.append(name)
        worst = max(worst, level)

    t = reading.airTempC
    if t is not None:
        if t < 32.0:
            flag("severeColdAir", 3)
        elif t < AIR_TEMP_MIN:
            flag("coldAir", 2)
        elif t > 39.0:
            flag("hotAir_critical", 3)
        elif t > AIR_TEMP_MAX:
            flag("hotAir", 2)

    h = reading.humidityPct
    if h is not None:
        if h < 40.0:
            flag("veryDryAir", 2)
        elif h < HUMIDITY_MIN:
            flag("dryAir", 1)
        elif h > 70.0:
            flag("veryHumidAir", 2)
        elif h > HUMIDITY_MAX:
            flag("humidAir", 1)

    p = reading.pressureHpa
    if p is not None:
        if p < PRESSURE_LOW:
            flag("lowPressure", 2)
        elif p > PRESSURE_HIGH:
            flag("highPressure", 1)

    aq = reading.airQualityPpm
    if aq is not None:
        if aq > AIR_QUALITY_CRIT:
            flag("poorAirQuality_critical", 3)
        elif aq > AIR_QUALITY_WARN:
            flag("poorAirQuality", 2)

    lux = reading.lightLux
    if lux is not None and lux > LIGHT_LUX_HIGH:
        flag("brightLight", 1)

    lid = reading.lidDistanceCm
    if lid is not None and lid > LID_OPEN_CM:
        flag("lidOpen", 2)

    current = reading.heaterCurrentA
    if current is not None:
        if current < HEATER_MIN_A and (t is not None and t < AIR_TEMP_MIN):
            flag("heaterFail", 3)
        elif current < HEATER_MIN_A:
            flag("heaterOff", 1)

    vib = _max_accel(reading)
    if vib > VIBRATION_G_CRIT:
        flag("severeVibration", 3)
    elif vib > VIBRATION_G_WARN:
        flag("vibration", 2)

    # --- Physiological Vital Rules (Neonatal) ---
    bpm = reading.bpm
    if bpm is not None:
        if bpm < 80:
            flag("severeBradycardia", 3)
        elif bpm < 100:
            flag("bradycardia", 2)
        elif bpm > 200:
            flag("severeTachycardia", 3)
        elif bpm > 170:
            flag("tachycardia", 2)

    spo2 = reading.spO2
    if spo2 is not None:
        if spo2 < 85:
            flag("severeHypoxemia", 3)
        elif spo2 < 92:
            flag("hypoxemia", 2)
        elif spo2 < 95:
            flag("lowSpO2", 1)

    sbp = reading.bloodPressureSystolic
    if sbp is not None:
        if sbp < 45:
            flag("severeHypotension", 3)
        elif sbp < 60:
            flag("hypotension", 2)

    severity_map: dict[int, Severity] = {0: "normal", 1: "watch", 2: "alert", 3: "critical"}
    return severity_map[worst], rules
