from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HistoryEvent:
    ts: str
    type: str
    message: str
    payload_json: str


def _connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def add_event(db_path: str, event_type: str, message: str, payload_json: str = "{}") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO history (ts, type, message, payload_json) VALUES (?, ?, ?, ?)",
            (ts, event_type, message, payload_json),
        )
        conn.commit()
    finally:
        conn.close()


def list_events(db_path: str, limit: int = 200, event_type: Optional[str] = None) -> List[HistoryEvent]:
    conn = _connect(db_path)
    try:
        if event_type:
            rows = conn.execute(
                "SELECT ts, type, message, payload_json FROM history WHERE type=? ORDER BY id DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ts, type, message, payload_json FROM history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [HistoryEvent(ts=r[0], type=r[1], message=r[2], payload_json=r[3]) for r in rows]
    finally:
        conn.close()
