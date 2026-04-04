from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class EventStore:
    def __init__(self, db_path: str = "data/cardsense.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    camera_id TEXT,
                    camera_type TEXT,
                    frame_id TEXT,
                    event_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    accepted INTEGER NOT NULL,
                    reason TEXT,
                    reason_code TEXT,
                    reason_detail TEXT,
                    state_version INTEGER,
                    transition_id TEXT,
                    state_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    mode TEXT,
                    best_card TEXT,
                    best_ev REAL,
                    confidence REAL,
                    rec_json TEXT NOT NULL
                )
                """
            )

    def log_event(self, game_id: str, ts_ms: int, camera_id: str, camera_type: str, frame_id: str, event: dict[str, Any]) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO events_raw (game_id, ts_ms, camera_id, camera_type, frame_id, event_json) VALUES (?, ?, ?, ?, ?, ?)",
                (game_id, int(ts_ms), camera_id, camera_type, frame_id, json.dumps(event)),
            )

    def log_transition(
        self,
        game_id: str,
        ts_ms: int,
        accepted: bool,
        reason: str,
        reason_code: str,
        reason_detail: str,
        state_version: int,
        transition_id: str,
        state: dict[str, Any],
    ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO state_transitions
                (game_id, ts_ms, accepted, reason, reason_code, reason_detail, state_version, transition_id, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    int(ts_ms),
                    1 if accepted else 0,
                    reason,
                    reason_code,
                    reason_detail,
                    int(state_version),
                    transition_id,
                    json.dumps(state),
                ),
            )

    def log_recommendation(self, game_id: str, ts_ms: int, rec: dict[str, Any]) -> None:
        best = rec.get("best_action", {}) if isinstance(rec, dict) else {}
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO recommendations (game_id, ts_ms, mode, best_card, best_ev, confidence, rec_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game_id,
                    int(ts_ms),
                    str(rec.get("recommendation_mode", "")),
                    str(best.get("card", "")),
                    float(best.get("ev", 0.0) or 0.0),
                    float(best.get("confidence", 0.0) or 0.0),
                    json.dumps(rec),
                ),
            )

    def get_events(self, game_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT event_json FROM events_raw WHERE game_id = ? ORDER BY id DESC LIMIT ?",
                (game_id, max(1, min(limit, 5000))),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                out.append(json.loads(str(row["event_json"])))
            except Exception:
                pass
        return out

    def replay_game(self, game_id: str, speed_multiplier: float = 1.0, limit: int = 1000) -> list[dict[str, Any]]:
        events = list(reversed(self.get_events(game_id, limit=limit)))
        if speed_multiplier <= 0:
            speed_multiplier = 1.0
        # return replay metadata; caller handles live emission pacing
        return events
