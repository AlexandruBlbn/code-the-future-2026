"""
Newton's Law of Cooling simulator for what-if scenarios.
dT/dt = k * (T_air - T_baby) + Q_heater / (m * c)
"""
from __future__ import annotations
import time
from .models import SensorReading, WhatIfRequest, WhatIfResponse
from .rules_engine import classify

# Physics constants (from clinical-rules.json)
K_CLOSED = 0.08   # thermal transfer coeff /min – lid closed
K_OPEN   = 0.25   # /min – lid open
BABY_MASS_KG = 3.2
SPECIFIC_HEAT = 3470.0   # J/(kg·K)
HEATER_POWER_W = 80.0
DT_SEC = 1.0  # Euler step


_PRESETS: dict[str, dict] = {
    "hyperthermia": {"airTempC": 40.0},
    "heaterFail":   {"heaterCurrentA": 0.0, "heaterActive": False},
    "lidOpen":      {"lidDistanceCm": 20.0, "lidOpen": True},
    "sensorFail":   {"airTempC": None},
    "ventBlocked":  {"servoAngleDeg": 0, "humidityPct": 75.0},
}


def _servo_to_k(servo_angle: int, lid_open: bool) -> float:
    """Higher servo angle = more ventilation = faster cooling."""
    k = K_OPEN if lid_open else K_CLOSED
    # Servo opens a vent: each 90° doubles effective cooling
    return k * (1.0 + servo_angle / 90.0)


def run_what_if(request: WhatIfRequest, base: SensorReading) -> WhatIfResponse:
    overrides = dict(request.overrides)
    if request.preset and request.preset in _PRESETS:
        overrides.update(_PRESETS[request.preset])

    # Clone base reading with overrides applied
    base_dict = base.model_dump()
    base_dict.update({k: v for k, v in overrides.items() if v is not None})

    T_air  = base_dict.get("airTempC") or 34.0
    T_baby = base_dict.get("babyTempC") or T_air  # start from current baby temp if available
    
    # Base vitals from reading or defaults
    bpm_base  = base_dict.get("bpm") or 130.0
    sys_base  = base_dict.get("bloodPressureSystolic") or 75.0
    dia_base  = base_dict.get("bloodPressureDiastolic") or 45.0
    spo2_base = base_dict.get("spO2") or 98.0

    heater_on = base_dict.get("heaterActive", True)
    lid_open  = base_dict.get("lidOpen", False)
    servo     = base_dict.get("servoAngleDeg") or 0

    k = _servo_to_k(servo, lid_open)
    Q = HEATER_POWER_W if heater_on else 0.0

    horizon = request.horizonSec
    trajectory: list[dict] = []
    time_to_risk: int | None = None
    ts_now = int(time.time())

    for step in range(horizon):
        # Euler integration (dt = 1s, k in /min → /60)
        dT = (k / 60.0) * (T_air - T_baby) + Q / (BABY_MASS_KG * SPECIFIC_HEAT)
        T_baby += dT

        # Couple vitals to T_baby
        # Bradycardia if < 35°C, Tachycardia if > 38°C
        t_dev_cold = max(0, 36.5 - T_baby)
        t_dev_hot  = max(0, T_baby - 37.5)
        
        sim_bpm  = bpm_base + (t_dev_hot * 15.0) - (t_dev_cold * 12.0)
        sim_sys  = sys_base - (t_dev_cold * 4.0) - (t_dev_hot * 2.0)
        sim_dia  = dia_base - (t_dev_cold * 2.0)
        sim_spo2 = spo2_base - (t_dev_cold * 1.5) - (t_dev_hot * 0.5)

        if step % 5 == 0:  # sample every 5s to keep payload small
            sim_reading = SensorReading(
                ts=ts_now + step,
                airTempC=T_air,
                humidityPct=base_dict.get("humidityPct"),
                heaterCurrentA=base_dict.get("heaterCurrentA"),
                heaterActive=heater_on,
                lidDistanceCm=base_dict.get("lidDistanceCm"),
                lidOpen=lid_open,
                servoAngleDeg=servo,
                bpm=round(sim_bpm, 1),
                bloodPressureSystolic=round(sim_sys, 1),
                bloodPressureDiastolic=round(sim_dia, 1),
                spO2=round(sim_spo2, 1)
            )
            severity, rules = classify(sim_reading)
            trajectory.append({
                "step": step,
                "ts": ts_now + step,
                "airTempC": round(T_air, 2),
                "babyTempC": round(T_baby, 2),
                "bpm": round(sim_bpm, 1),
                "sys": round(sim_sys, 1),
                "dia": round(sim_dia, 1),
                "spO2": round(sim_spo2, 1),
                "severity": severity,
                "rules": rules,
            })

        if time_to_risk is None and (T_baby < 36.0 or T_baby > 38.5 or sim_bpm < 100 or sim_spo2 < 92):
            time_to_risk = step

    # Summary sentence
    final_baby = round(T_baby, 1)
    if time_to_risk is not None:
        summary = f"Clinical risk detected in {time_to_risk}s. Final baby temp: {final_baby}°C."
    else:
        summary = f"No immediate clinical risk in {horizon}s window."

    return WhatIfResponse(
        trajectory=trajectory,
        summary=summary,
        timeToRiskSec=time_to_risk,
    )
