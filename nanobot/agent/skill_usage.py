"""Skill usage tracking: per-skill state, activity timestamps, and counters.

Lightweight SQLite-backed store that lives alongside the memory DB
(``workspace/memory/history.db``) in its own ``skill_usage`` table. Data
is keyed by skill name; rows are created on first observation.

Used by the curator (state transitions) and by ``manage_skill`` (create /
edit / patch / delete打点).

This module is intentionally independent of ``MemoryStore`` to avoid
import cycles and to let callers use it without a full memory pipeline.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"

_VALID_STATES = {STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillUsageStore:
    """Per-workspace skill usage store backed by the memory SQLite DB.

    Concurrency: sqlite3 connection is opened with ``check_same_thread=False``
    and access is serialized by an instance lock — mirrors ``MemoryStore``'s
    pattern so the two stores can coexist on the same file.
    """

    def __init__(self, workspace_dir: Path):
        self._db_path = workspace_dir / "memory" / "history.db"
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    # -- connection ---------------------------------------------------------

    def _get_db(self) -> sqlite3.Connection:
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._init_db()
        return self._db

    def _init_db(self) -> None:
        assert self._db is not None
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS skill_usage (
                name             TEXT PRIMARY KEY,
                state            TEXT NOT NULL DEFAULT 'active',
                created_at       TEXT NOT NULL,
                last_activity_at TEXT,
                view_count       INTEGER NOT NULL DEFAULT 0,
                use_count        INTEGER NOT NULL DEFAULT 0,
                patch_count      INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_skill_usage_state
                ON skill_usage(state);
            """
        )
        self._db.commit()

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                try:
                    self._db.close()
                except sqlite3.Error:
                    pass
                self._db = None

    # -- row access ---------------------------------------------------------

    def get_record(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            db = self._get_db()
            row = db.execute(
                "SELECT name, state, created_at, last_activity_at, "
                "view_count, use_count, patch_count "
                "FROM skill_usage WHERE name = ?",
                (name,),
            ).fetchone()
        return dict(row) if row else None

    def _ensure_row(self, name: str, *, now: str | None = None) -> None:
        """Insert a default row if it doesn't exist. No-op otherwise."""
        ts = now or _now_iso()
        db = self._get_db()
        db.execute(
            "INSERT OR IGNORE INTO skill_usage "
            "(name, state, created_at, last_activity_at) "
            "VALUES (?, ?, ?, ?)",
            (name, STATE_ACTIVE, ts, ts),
        )

    # -- mutations ----------------------------------------------------------

    def record_create(self, name: str) -> None:
        """Record that a skill was freshly created."""
        ts = _now_iso()
        with self._lock:
            db = self._get_db()
            # Use REPLACE semantics: recreating a previously-archived skill
            # resets its state to active and zeros counters. This matches the
            # user's mental model — "it's a new skill now".
            db.execute(
                "INSERT OR REPLACE INTO skill_usage "
                "(name, state, created_at, last_activity_at, "
                "view_count, use_count, patch_count) "
                "VALUES (?, ?, ?, ?, 0, 0, 0)",
                (name, STATE_ACTIVE, ts, ts),
            )
            db.commit()

    def bump_view(self, name: str) -> None:
        self._bump(name, "view_count")

    def bump_use(self, name: str) -> None:
        self._bump(name, "use_count")

    def bump_patch(self, name: str) -> None:
        self._bump(name, "patch_count")

    def _bump(self, name: str, column: str) -> None:
        if column not in {"view_count", "use_count", "patch_count"}:
            raise ValueError(f"invalid counter column: {column!r}")
        ts = _now_iso()
        with self._lock:
            db = self._get_db()
            self._ensure_row(name, now=ts)
            db.execute(
                f"UPDATE skill_usage "
                f"SET {column} = {column} + 1, last_activity_at = ? "
                f"WHERE name = ?",
                (ts, name),
            )
            db.commit()

    def set_state(self, name: str, state: str) -> None:
        if state not in _VALID_STATES:
            raise ValueError(f"invalid state: {state!r}")
        with self._lock:
            db = self._get_db()
            self._ensure_row(name)
            db.execute(
                "UPDATE skill_usage SET state = ? WHERE name = ?",
                (state, name),
            )
            db.commit()

    def forget(self, name: str) -> None:
        """Drop the usage record. Used when a skill is hard-deleted."""
        with self._lock:
            db = self._get_db()
            db.execute("DELETE FROM skill_usage WHERE name = ?", (name,))
            db.commit()

    # -- reporting ----------------------------------------------------------

    def all_records(self) -> list[dict[str, Any]]:
        with self._lock:
            db = self._get_db()
            rows = db.execute(
                "SELECT name, state, created_at, last_activity_at, "
                "view_count, use_count, patch_count "
                "FROM skill_usage ORDER BY name"
            ).fetchall()
        return [dict(r) for r in rows]

    def least_recently_used(self, limit: int = 5) -> list[dict[str, Any]]:
        """Return the N skills with the oldest last_activity_at."""
        with self._lock:
            db = self._get_db()
            rows = db.execute(
                "SELECT name, state, last_activity_at, use_count "
                "FROM skill_usage "
                "WHERE state != ? "
                "ORDER BY last_activity_at ASC NULLS FIRST "
                "LIMIT ?",
                (STATE_ARCHIVED, limit),
            ).fetchall()
        return [dict(r) for r in rows]
