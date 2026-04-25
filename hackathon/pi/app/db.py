"""
SQLite ring-buffer DB + in-memory deque for fast reads.
Stores last 300 seconds of SensorReading rows.
"""
from __future__ import annotations
import sqlite3
import json
import time
from collections import deque
from threading import Lock
from pathlib import Path

from .models import SensorReading

DB_PATH = Path(__file__).parent.parent / "data" / "neotwin.db"
RING_SIZE = 300  # seconds

# In-memory ring buffer for fast SSE/history reads
_ring: deque[dict] = deque(maxlen=RING_SIZE)
_lock = Lock()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS readings (
            ts INTEGER PRIMARY KEY,
            payload TEXT NOT NULL
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON readings(ts)")
    con.commit()
    con.close()


def insert(reading: SensorReading) -> None:
    row = reading.model_dump()
    with _lock:
        _ring.append(row)

    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO readings(ts, payload) VALUES (?, ?)",
        (reading.ts, json.dumps(row)),
    )
    # Prune rows older than 1 hour
    con.execute("DELETE FROM readings WHERE ts < ?", (int(time.time()) - 3600,))
    con.commit()
    con.close()


def get_latest() -> dict | None:
    with _lock:
        return _ring[-1] if _ring else None


def get_history(seconds: int = 60) -> list[dict]:
    cutoff = int(time.time()) - seconds
    with _lock:
        return [r for r in _ring if r.get("ts", 0) >= cutoff]


def seed_ring_from_db() -> None:
    """Load last RING_SIZE rows from SQLite into memory on startup."""
    if not DB_PATH.exists():
        return
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT payload FROM readings ORDER BY ts DESC LIMIT ?", (RING_SIZE,)
    ).fetchall()
    con.close()
    with _lock:
        for (payload,) in reversed(rows):
            _ring.append(json.loads(payload))
