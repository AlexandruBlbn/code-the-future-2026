"""
FastAPI route handlers:
  POST /ingest        – ESP32 pushes sensor data
  GET  /state         – latest TwinState
  GET  /history       – ring-buffer slice
  GET  /stream        – SSE live feed
  POST /simulate      – what-if scenarios
  GET  /rules         – clinical rules JSON
  POST /servo         – override servo angle (frontend command)
"""
from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .models import (
    SensorReading, TwinState, WhatIfRequest,
    WhatIfResponse, IngestResponse, Prediction,
)
from .rules_engine import classify
from .simulator import run_what_if
from . import db

router = APIRouter()

# Shared servo command (backend → ESP32 via /ingest response)
_servo_command: int = 0

# SSE subscriber queues
_subscribers: list[asyncio.Queue] = []
_frontend_status_task: asyncio.Task | None = None


async def _log_frontend_connection_status() -> None:
    while True:
        if _subscribers:
            print("frontend e conectat la resebery pi.")
        else:
            print("astept conexiunea cu frontend")
        await asyncio.sleep(5)


def start_frontend_connection_logger() -> None:
    global _frontend_status_task
    if _frontend_status_task is None or _frontend_status_task.done():
        _frontend_status_task = asyncio.create_task(_log_frontend_connection_status())


def _compute_prediction(reading: SensorReading) -> Prediction:
    """Quick 1/5/15-min thermal prediction (Newton cooling, simplified)."""
    T = reading.airTempC or 34.0
    k = 0.08 if not reading.lidOpen else 0.25
    heater_on = reading.heaterActive or False
    Q = 80.0 if heater_on else 0.0
    m, c = 3.2, 3470.0

    def project(minutes: float) -> float:
        steps = int(minutes * 60)
        tmp = T
        for _ in range(steps):
            tmp += (k / 60.0) * (T - tmp) + Q / (m * c)
        return round(tmp, 2)

    return Prediction(t1=project(1), t5=project(5), t15=project(15))


async def _broadcast(state: TwinState) -> None:
    data = "data: " + state.model_dump_json() + "\n\n"
    dead = []
    for q in _subscribers:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _subscribers.remove(q)


# ─── POST /ingest ─────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest(reading: SensorReading) -> IngestResponse:
    if reading.ts == 0:
        reading.ts = int(time.time())

    db.insert(reading)
    severity, rules = classify(reading)
    prediction = _compute_prediction(reading)

    state = TwinState(
        reading=reading,
        severity=severity,
        activeRules=rules,
        prediction=prediction,
        servoCommand=_servo_command,
    )
    asyncio.create_task(_broadcast(state))
    return IngestResponse(ok=True, servoAngleDeg=_servo_command)


# ─── GET /state ───────────────────────────────────────────────────────────────

@router.get("/state")
async def state() -> dict:
    latest = db.get_latest()
    if not latest:
        raise HTTPException(status_code=404, detail="No data yet")
    reading = SensorReading(**latest)
    severity, rules = classify(reading)
    prediction = _compute_prediction(reading)
    return TwinState(
        reading=reading,
        severity=severity,
        activeRules=rules,
        prediction=prediction,
        servoCommand=_servo_command,
    ).model_dump()


# ─── GET /history ─────────────────────────────────────────────────────────────

@router.get("/history")
async def history(seconds: int = 60) -> list[dict]:
    return db.get_history(seconds)


# ─── GET /stream (SSE) ────────────────────────────────────────────────────────

@router.get("/stream")
async def stream() -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    _subscribers.append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield data
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if queue in _subscribers:
                _subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── POST /simulate ───────────────────────────────────────────────────────────

@router.post("/simulate", response_model=WhatIfResponse)
async def simulate(request: WhatIfRequest) -> WhatIfResponse:
    latest = db.get_latest()
    if not latest:
        # Use neutral defaults if no real data
        base = SensorReading(
            ts=int(time.time()),
            airTempC=34.0,
            humidityPct=55.0,
            heaterCurrentA=0.5,
            heaterActive=True,
            lidDistanceCm=2.0,
            lidOpen=False,
            servoAngleDeg=0,
        )
    else:
        base = SensorReading(**latest)
    return run_what_if(request, base)


# ─── GET /rules ───────────────────────────────────────────────────────────────

@router.get("/rules")
async def rules() -> dict:
    rules_path = Path(__file__).parent.parent / "data" / "clinical-rules.json"
    if not rules_path.exists():
        raise HTTPException(status_code=404, detail="clinical-rules.json not found")
    return json.loads(rules_path.read_text())


# ─── POST /servo ──────────────────────────────────────────────────────────────

@router.post("/servo")
async def set_servo(angle: int) -> dict:
    global _servo_command
    _servo_command = max(0, min(180, angle))
    return {"ok": True, "servoAngleDeg": _servo_command}
