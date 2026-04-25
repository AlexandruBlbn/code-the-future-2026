"""
NeoTwin FastAPI application — entry point.
Run: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router, start_frontend_connection_logger
from . import db

app = FastAPI(
    title="NeoTwin API",
    version="1.0.0",
    description="Digital Twin backend for neonatal incubator monitoring",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup() -> None:
    db.init_db()
    db.seed_ring_from_db()
    start_frontend_connection_logger()
    print("[NeoTwin] DB ready. Waiting for ESP32 data or mock input.")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
