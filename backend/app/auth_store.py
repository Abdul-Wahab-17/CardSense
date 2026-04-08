from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _hash_password(password_plain: str, salt: bytes | None = None) -> tuple[str, str]:
    s = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password_plain.encode("utf-8"), s, 150000)
    return s.hex(), dk.hex()


def _verify_password(password_plain: str, salt_hex: str, digest_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    _, calc = _hash_password(password_plain, salt=salt)
    return hmac.compare_digest(calc, digest_hex)


class AuthStore:
    def __init__(self, db_path: str = "data/cardsense_auth.db") -> None:
        self.db_path = db_path
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
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    display_name TEXT,
                    status TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_login_at INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS roles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    UNIQUE(user_id, role)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token_hash TEXT UNIQUE NOT NULL,
                    expires_at INTEGER NOT NULL,
                    revoked_at INTEGER,
                    ip TEXT,
                    user_agent TEXT,
                    session_id TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    key_hash TEXT UNIQUE NOT NULL,
                    label TEXT,
                    scopes TEXT,
                    revoked_at INTEGER
                )
                """
            )

    def create_user_account(self, email: str, password_plain: str, display_name: str) -> dict[str, Any]:
        now = int(time.time())
        salt_hex, digest_hex = _hash_password(password_plain)
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, password_salt, display_name, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (email.lower().strip(), digest_hex, salt_hex, display_name.strip(), "active", now),
            )
            user_id = int(cur.lastrowid)
            conn.execute("INSERT OR IGNORE INTO roles (user_id, role) VALUES (?, ?)", (user_id, "operator"))
        return {"user_id": user_id, "created_at": now}

    def authenticate_user(self, email: str, password_plain: str) -> dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
            if row is None:
                return {"auth_ok": False}
            if str(row["status"]) != "active":
                return {"auth_ok": False}
            ok = _verify_password(password_plain, str(row["password_salt"]), str(row["password_hash"]))
            if not ok:
                return {"auth_ok": False}
            user_id = int(row["id"])
            conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (int(time.time()), user_id))
            roles = [str(r[0]) for r in conn.execute("SELECT role FROM roles WHERE user_id = ?", (user_id,)).fetchall()]
            return {
                "auth_ok": True,
                "user": {
                    "id": user_id,
                    "email": str(row["email"]),
                    "display_name": str(row["display_name"] or ""),
                    "status": str(row["status"]),
                },
                "roles": roles,
            }

    def get_user_and_roles(self, user_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
            if row is None:
                return None
            roles = [str(r[0]) for r in conn.execute("SELECT role FROM roles WHERE user_id = ?", (int(user_id),)).fetchall()]
            return {
                "user": {
                    "id": int(row["id"]),
                    "email": str(row["email"]),
                    "display_name": str(row["display_name"] or ""),
                    "status": str(row["status"]),
                },
                "roles": roles,
            }

    def issue_refresh_token(self, user_id: int, session_id: str, ttl_sec: int, client_meta: dict[str, str]) -> dict[str, str]:
        token = secrets.token_urlsafe(48)
        token_hash = _hash_secret(token)
        expires_at = int(time.time()) + max(int(ttl_sec), 60)
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO refresh_tokens (user_id, token_hash, expires_at, revoked_at, ip, user_agent, session_id) VALUES (?, ?, ?, NULL, ?, ?, ?)",
                (user_id, token_hash, expires_at, client_meta.get("ip", ""), client_meta.get("user_agent", ""), session_id),
            )
            rid = int(cur.lastrowid)
        return {"refresh_token": token, "refresh_id": str(rid), "expires_at": str(expires_at)}

    def rotate_refresh_token(self, refresh_token_old: str, ttl_sec: int, client_meta: dict[str, str]) -> dict[str, Any]:
        old_hash = _hash_secret(refresh_token_old)
        now = int(time.time())
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM refresh_tokens WHERE token_hash = ?", (old_hash,)).fetchone()
            if row is None:
                return {"ok": False}
            if row["revoked_at"] is not None or int(row["expires_at"]) < now:
                return {"ok": False}
            user_row = conn.execute("SELECT status FROM users WHERE id = ?", (int(row["user_id"]),)).fetchone()
            if user_row is None or str(user_row["status"]) != "active":
                return {"ok": False}
            conn.execute("UPDATE refresh_tokens SET revoked_at = ? WHERE id = ?", (now, int(row["id"])))
            token = secrets.token_urlsafe(48)
            token_hash = _hash_secret(token)
            expires_at = int(time.time()) + max(int(ttl_sec), 60)
            cur = conn.execute(
                "INSERT INTO refresh_tokens (user_id, token_hash, expires_at, revoked_at, ip, user_agent, session_id) VALUES (?, ?, ?, NULL, ?, ?, ?)",
                (
                    int(row["user_id"]),
                    token_hash,
                    expires_at,
                    client_meta.get("ip", ""),
                    client_meta.get("user_agent", ""),
                    str(row["session_id"]),
                ),
            )
            rid = int(cur.lastrowid)
            return {
                "ok": True,
                "user_id": int(row["user_id"]),
                "session_id": str(row["session_id"]),
                "refresh_token": token,
                "refresh_id": str(rid),
                "expires_at": str(expires_at),
            }

    def revoke_session_tokens(self, session_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE refresh_tokens SET revoked_at = ? WHERE session_id = ? AND revoked_at IS NULL",
                (int(time.time()), str(session_id)),
            )
            return int(cur.rowcount or 0)

    def set_user_status(self, user_id: int, status: str) -> bool:
        status_norm = str(status).strip().lower()
        if status_norm not in {"active", "blocked"}:
            return False
        with self._conn() as conn:
            cur = conn.execute("UPDATE users SET status = ? WHERE id = ?", (status_norm, int(user_id)))
            return int(cur.rowcount or 0) > 0

    def assign_role(self, user_id: int, role: str) -> bool:
        role_norm = str(role).strip()
        if not role_norm:
            return False
        with self._conn() as conn:
            conn.execute("INSERT OR IGNORE INTO roles (user_id, role) VALUES (?, ?)", (int(user_id), role_norm))
        return True

    def revoke_refresh_token(self, refresh_token: str) -> bool:
        token_hash = _hash_secret(refresh_token)
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM refresh_tokens WHERE token_hash = ?", (token_hash,)).fetchone()
            if row is None:
                return False
            conn.execute("UPDATE refresh_tokens SET revoked_at = ? WHERE id = ?", (int(time.time()), int(row[0])))
            return True
