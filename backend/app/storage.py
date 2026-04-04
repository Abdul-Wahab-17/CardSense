from __future__ import annotations

import json
import sqlite3
import threading
import importlib
from pathlib import Path
from typing import Any

from .config import settings


class EventStore:
    def __init__(self, db_path: str = "data/cardsense.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self.database_url = settings.database_url.strip() or ""
        self._use_sqlite = not self.database_url
        if self._use_sqlite:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not self._use_sqlite:
            try:
                psycopg = importlib.import_module("psycopg")
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"psycopg is required for DATABASE_URL usage: {exc}")
            conn = psycopg.connect(
                self.database_url,
                connect_timeout=int(settings.db_connect_timeout_ms) // 1000,
            )
            return conn
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _param(self) -> str:
        return "?" if self._use_sqlite else "%s"

    def _sql(self, template: str) -> str:
        return template.format(p=self._param())

    def _init_schema(self) -> None:
        with self._conn() as conn:
            if not self._use_sqlite:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events_raw (
                        id BIGSERIAL PRIMARY KEY,
                        game_id TEXT NOT NULL,
                        ts_ms BIGINT NOT NULL,
                        camera_id TEXT,
                        camera_type TEXT,
                        frame_id TEXT,
                        event_json TEXT NOT NULL,
                        idempotency_event_key TEXT UNIQUE
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS state_transitions (
                        id BIGSERIAL PRIMARY KEY,
                        game_id TEXT NOT NULL,
                        ts_ms BIGINT NOT NULL,
                        accepted INTEGER NOT NULL,
                        reason TEXT,
                        reason_code TEXT,
                        reason_detail TEXT,
                        state_version INTEGER,
                        transition_id TEXT,
                        state_json TEXT NOT NULL,
                        idempotency_transition_key TEXT UNIQUE
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS recommendations (
                        id BIGSERIAL PRIMARY KEY,
                        game_id TEXT NOT NULL,
                        ts_ms BIGINT NOT NULL,
                        mode TEXT,
                        best_card TEXT,
                        best_ev DOUBLE PRECISION,
                        confidence DOUBLE PRECISION,
                        rec_json TEXT NOT NULL,
                        idempotency_recommendation_key TEXT UNIQUE
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS game_configs (
                        id BIGSERIAL PRIMARY KEY,
                        game_id TEXT NOT NULL,
                        ts_ms BIGINT NOT NULL,
                        config_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reducer_offsets (
                        id BIGSERIAL PRIMARY KEY,
                        consumer_group TEXT NOT NULL,
                        stream_name TEXT NOT NULL,
                        last_message_id TEXT NOT NULL,
                        updated_ts_ms BIGINT NOT NULL,
                        UNIQUE(consumer_group, stream_name)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS deploy_audit_log (
                        id BIGSERIAL PRIMARY KEY,
                        ts_ms BIGINT NOT NULL,
                        deploy_env TEXT,
                        release_version TEXT,
                        details_json TEXT
                    )
                    """
                )
                return
            def _has_column(table: str, column: str) -> bool:
                if not self._use_sqlite:
                    rows = conn.execute(
                        "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                        (table,),
                    ).fetchall()
                    return any(str(r[0]) == column for r in rows)
                rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
                return any(str(r[1]) == column for r in rows)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    camera_id TEXT,
                    camera_type TEXT,
                    frame_id TEXT,
                    event_json TEXT NOT NULL,
                    idempotency_event_key TEXT
                )
                """
            )
            if not _has_column("events_raw", "idempotency_event_key"):
                try:
                    conn.execute("ALTER TABLE events_raw ADD COLUMN idempotency_event_key TEXT")
                except Exception:
                    pass
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_events_event_key ON events_raw(idempotency_event_key)")
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
                    state_json TEXT NOT NULL,
                    idempotency_transition_key TEXT
                )
                """
            )
            if not _has_column("state_transitions", "idempotency_transition_key"):
                try:
                    conn.execute("ALTER TABLE state_transitions ADD COLUMN idempotency_transition_key TEXT")
                except Exception:
                    pass
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_transition_key ON state_transitions(idempotency_transition_key)")
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
                    rec_json TEXT NOT NULL,
                    idempotency_recommendation_key TEXT
                )
                """
            )
            if not _has_column("recommendations", "idempotency_recommendation_key"):
                try:
                    conn.execute("ALTER TABLE recommendations ADD COLUMN idempotency_recommendation_key TEXT")
                except Exception:
                    pass
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_recommendation_key ON recommendations(idempotency_recommendation_key)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS game_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    config_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reducer_offsets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    consumer_group TEXT NOT NULL,
                    stream_name TEXT NOT NULL,
                    last_message_id TEXT NOT NULL,
                    updated_ts_ms INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_reducer_offsets_group_stream ON reducer_offsets(consumer_group, stream_name)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS deploy_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER NOT NULL,
                    deploy_env TEXT,
                    release_version TEXT,
                    details_json TEXT
                )
                """
            )

    def log_event(self, game_id: str, ts_ms: int, camera_id: str, camera_type: str, frame_id: str, event: dict[str, Any]) -> None:
        idem = f"{game_id}:{frame_id}" if frame_id else None
        with self._lock, self._conn() as conn:
            sql = self._sql(
                "INSERT INTO events_raw (game_id, ts_ms, camera_id, camera_type, frame_id, event_json, idempotency_event_key) "
                "VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}) "
            )
            if self._use_sqlite:
                sql = sql.replace("INSERT INTO", "INSERT OR IGNORE INTO")
            else:
                sql = sql + "ON CONFLICT (idempotency_event_key) DO NOTHING"
            conn.execute(sql, (game_id, int(ts_ms), camera_id, camera_type, frame_id, json.dumps(event), idem))

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
        idem = f"{game_id}:{transition_id}" if transition_id else None
        with self._lock, self._conn() as conn:
            sql = self._sql(
                "INSERT INTO state_transitions "
                "(game_id, ts_ms, accepted, reason, reason_code, reason_detail, state_version, transition_id, state_json, idempotency_transition_key) "
                "VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}) "
            )
            if self._use_sqlite:
                sql = sql.replace("INSERT INTO", "INSERT OR IGNORE INTO")
            else:
                sql = sql + "ON CONFLICT (idempotency_transition_key) DO NOTHING"
            conn.execute(
                sql,
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
                    idem,
                ),
            )

    def log_recommendation(self, game_id: str, ts_ms: int, rec: dict[str, Any]) -> None:
        best = rec.get("best_action", {}) if isinstance(rec, dict) else {}
        idem = f"{game_id}:{int(rec.get('freshness_ts_ms', ts_ms) or ts_ms)}:{str(best.get('card', ''))}"
        with self._lock, self._conn() as conn:
            sql = self._sql(
                "INSERT INTO recommendations "
                "(game_id, ts_ms, mode, best_card, best_ev, confidence, rec_json, idempotency_recommendation_key) "
                "VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}) "
            )
            if self._use_sqlite:
                sql = sql.replace("INSERT INTO", "INSERT OR IGNORE INTO")
            else:
                sql = sql + "ON CONFLICT (idempotency_recommendation_key) DO NOTHING"
            conn.execute(
                sql,
                (
                    game_id,
                    int(ts_ms),
                    str(rec.get("recommendation_mode", "")),
                    str(best.get("card", "")),
                    float(best.get("ev", 0.0) or 0.0),
                    float(best.get("confidence", 0.0) or 0.0),
                    json.dumps(rec),
                    idem,
                ),
            )

    def log_transition_and_recommendation(
        self,
        game_id: str,
        ts_ms: int,
        transition: dict[str, Any],
        recommendation: dict[str, Any],
    ) -> None:
        with self._lock, self._conn() as conn:
            try:
                if not self._use_sqlite:
                    conn.execute("BEGIN")

                t_idem = f"{game_id}:{str(transition.get('transition_id', ''))}"
                t_sql = self._sql(
                    "INSERT INTO state_transitions "
                    "(game_id, ts_ms, accepted, reason, reason_code, reason_detail, state_version, transition_id, state_json, idempotency_transition_key) "
                    "VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}) "
                )
                if self._use_sqlite:
                    t_sql = t_sql.replace("INSERT INTO", "INSERT OR IGNORE INTO")
                else:
                    t_sql = t_sql + "ON CONFLICT (idempotency_transition_key) DO NOTHING"
                conn.execute(
                    t_sql,
                    (
                        game_id,
                        int(ts_ms),
                        1 if bool(transition.get("accepted")) else 0,
                        str(transition.get("reason")),
                        str(transition.get("reason_code")),
                        str(transition.get("reason_detail")),
                        int(transition.get("state_version", 0)),
                        str(transition.get("transition_id", "")),
                        json.dumps(dict(transition.get("state", {}))),
                        t_idem,
                    ),
                )

                best = recommendation.get("best_action", {}) if isinstance(recommendation, dict) else {}
                r_idem = f"{game_id}:{int(recommendation.get('freshness_ts_ms', ts_ms) or ts_ms)}:{str(best.get('card', ''))}"
                r_sql = self._sql(
                    "INSERT INTO recommendations "
                    "(game_id, ts_ms, mode, best_card, best_ev, confidence, rec_json, idempotency_recommendation_key) "
                    "VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}) "
                )
                if self._use_sqlite:
                    r_sql = r_sql.replace("INSERT INTO", "INSERT OR IGNORE INTO")
                else:
                    r_sql = r_sql + "ON CONFLICT (idempotency_recommendation_key) DO NOTHING"
                conn.execute(
                    r_sql,
                    (
                        game_id,
                        int(ts_ms),
                        str(recommendation.get("recommendation_mode", "")),
                        str(best.get("card", "")),
                        float(best.get("ev", 0.0) or 0.0),
                        float(best.get("confidence", 0.0) or 0.0),
                        json.dumps(recommendation),
                        r_idem,
                    ),
                )

                if not self._use_sqlite:
                    conn.execute("COMMIT")
            except Exception:
                if not self._use_sqlite:
                    conn.execute("ROLLBACK")
                raise

    def log_game_config(self, game_id: str, ts_ms: int, config: dict[str, Any]) -> None:
        with self._lock, self._conn() as conn:
            sql = self._sql("INSERT INTO game_configs (game_id, ts_ms, config_json) VALUES ({p}, {p}, {p})")
            conn.execute(sql, (game_id, int(ts_ms), json.dumps(config)))

    def upsert_reducer_offset(self, consumer_group: str, stream_name: str, last_message_id: str, updated_ts_ms: int) -> None:
        with self._lock, self._conn() as conn:
            sql = self._sql(
                "INSERT INTO reducer_offsets (consumer_group, stream_name, last_message_id, updated_ts_ms) "
                "VALUES ({p}, {p}, {p}, {p}) "
                "ON CONFLICT(consumer_group, stream_name) "
                "DO UPDATE SET last_message_id = excluded.last_message_id, updated_ts_ms = excluded.updated_ts_ms"
            )
            conn.execute(sql, (consumer_group, stream_name, last_message_id, int(updated_ts_ms)))

    def log_deploy_audit(self, ts_ms: int, deploy_env: str, release_version: str, details: dict[str, Any]) -> None:
        with self._lock, self._conn() as conn:
            sql = self._sql("INSERT INTO deploy_audit_log (ts_ms, deploy_env, release_version, details_json) VALUES ({p}, {p}, {p}, {p})")
            conn.execute(sql, (int(ts_ms), deploy_env, release_version, json.dumps(details)))

    def purge_older_than_days(self, events_days: int, rec_days: int) -> dict[str, int]:
        now_ms = int(__import__("time").time() * 1000)
        event_cutoff = now_ms - (max(0, int(events_days)) * 24 * 60 * 60 * 1000)
        rec_cutoff = now_ms - (max(0, int(rec_days)) * 24 * 60 * 60 * 1000)
        with self._lock, self._conn() as conn:
            e_sql = self._sql("DELETE FROM events_raw WHERE ts_ms < {p}")
            t_sql = self._sql("DELETE FROM state_transitions WHERE ts_ms < {p}")
            r_sql = self._sql("DELETE FROM recommendations WHERE ts_ms < {p}")
            e = conn.execute(e_sql, (event_cutoff,)).rowcount
            t = conn.execute(t_sql, (event_cutoff,)).rowcount
            r = conn.execute(r_sql, (rec_cutoff,)).rowcount
        return {"events_raw": int(e or 0), "state_transitions": int(t or 0), "recommendations": int(r or 0)}

    def get_events(self, game_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._conn() as conn:
            sql = self._sql("SELECT event_json FROM events_raw WHERE game_id = {p} ORDER BY id DESC LIMIT {p}")
            rows = conn.execute(sql, (game_id, max(1, min(limit, 5000)))).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                out.append(json.loads(str(row[0])))
            except Exception:
                pass
        return out

    def replay_game(self, game_id: str, speed_multiplier: float = 1.0, limit: int = 1000) -> list[dict[str, Any]]:
        events = list(reversed(self.get_events(game_id, limit=limit)))
        if speed_multiplier <= 0:
            speed_multiplier = 1.0
        # return replay metadata; caller handles live emission pacing
        return events

    def export_game(self, game_id: str, limit: int = 5000) -> dict[str, Any]:
        with self._conn() as conn:
            ev_sql = self._sql("SELECT event_json FROM events_raw WHERE game_id = {p} ORDER BY id DESC LIMIT {p}")
            tr_sql = self._sql("SELECT state_json, accepted, reason_code, reason_detail FROM state_transitions WHERE game_id = {p} ORDER BY id DESC LIMIT {p}")
            rec_sql = self._sql("SELECT rec_json FROM recommendations WHERE game_id = {p} ORDER BY id DESC LIMIT {p}")
            events = conn.execute(ev_sql, (game_id, limit)).fetchall()
            transitions = conn.execute(tr_sql, (game_id, limit)).fetchall()
            recs = conn.execute(rec_sql, (game_id, limit)).fetchall()
        return {
            "game_id": game_id,
            "events": [json.loads(str(r[0])) for r in events],
            "transitions": [
                {
                    "state": json.loads(str(r[0])),
                    "accepted": bool(r[1]),
                    "reason_code": str(r[2]),
                    "reason_detail": str(r[3]),
                }
                for r in transitions
            ],
            "recommendations": [json.loads(str(r[0])) for r in recs],
        }

    def delete_game(self, game_id: str) -> dict[str, int]:
        with self._lock, self._conn() as conn:
            ev_sql = self._sql("DELETE FROM events_raw WHERE game_id = {p}")
            tr_sql = self._sql("DELETE FROM state_transitions WHERE game_id = {p}")
            rec_sql = self._sql("DELETE FROM recommendations WHERE game_id = {p}")
            cfg_sql = self._sql("DELETE FROM game_configs WHERE game_id = {p}")
            e = conn.execute(ev_sql, (game_id,)).rowcount
            t = conn.execute(tr_sql, (game_id,)).rowcount
            r = conn.execute(rec_sql, (game_id,)).rowcount
            c = conn.execute(cfg_sql, (game_id,)).rowcount
        return {"events_raw": int(e or 0), "state_transitions": int(t or 0), "recommendations": int(r or 0), "game_configs": int(c or 0)}
