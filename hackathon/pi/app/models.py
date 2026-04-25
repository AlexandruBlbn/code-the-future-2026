from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    ts: int = Field(..., description="UNIX timestamp (seconds)")

    # DHT22
    airTempC: Optional[float] = None
    humidityPct: Optional[float] = None

    # BMP280
    pressureHpa: Optional[float] = None
    altitudeM: Optional[float] = None
    bmpTempC: Optional[float] = None

    # MQ-135
    airQualityRaw: Optional[int] = None
    airQualityPpm: Optional[float] = None

    # LDR
    lightRaw: Optional[int] = None
    lightLux: Optional[float] = None

    # HC-SR04
    lidDistanceCm: Optional[float] = None
    lidOpen: Optional[bool] = None

    # MPU6050
    accelX: Optional[float] = None
    accelY: Optional[float] = None
    accelZ: Optional[float] = None
    gyroX: Optional[float] = None
    gyroY: Optional[float] = None
    gyroZ: Optional[float] = None
    mpuTempC: Optional[float] = None

    # ACS712
    heaterCurrentA: Optional[float] = None
    heaterActive: Optional[bool] = None

    # Physiological (simulated)
    bpm: Optional[float] = None
    bloodPressureSystolic: Optional[float] = None
    bloodPressureDiastolic: Optional[float] = None
    spO2: Optional[float] = None

    # SG90 servo (current reported angle)
    servoAngleDeg: Optional[int] = 0


Severity = Literal["normal", "watch", "alert", "critical"]


class Prediction(BaseModel):
    t1: float
    t5: float
    t15: float


class TwinState(BaseModel):
    reading: SensorReading
    severity: Severity
    activeRules: list[str]
    prediction: Prediction
    servoCommand: int = 0


class WhatIfRequest(BaseModel):
    overrides: dict = Field(default_factory=dict)
    horizonSec: int = 300
    preset: Optional[
        Literal["hyperthermia", "heaterFail", "lidOpen", "sensorFail", "ventBlocked"]
    ] = None


class WhatIfResponse(BaseModel):
    trajectory: list[dict]
    summary: str
    timeToRiskSec: Optional[int] = None


class IngestResponse(BaseModel):
    ok: bool = True
    servoAngleDeg: int = 0
