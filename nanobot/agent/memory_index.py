"""MemoryIndex — SQLite sidecar index for MEMORY.md."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_items (
  id            TEXT PRIMARY KEY,
  source_file   TEXT NOT NULL,
  section_path  TEXT NOT NULL,
  item_type     TEXT NOT NULL,
  content       TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  embedding     BLOB,
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL,
  removed_at    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_items_hash ON memory_items(content_hash);
CREATE INDEX IF NOT EXISTS idx_items_section ON memory_items(section_path);

CREATE TABLE IF NOT EXISTS concepts (
  id               TEXT PRIMARY KEY,
  name             TEXT NOT NULL,
  description      TEXT,
  centroid_embed   BLOB,
  member_count     INTEGER NOT NULL DEFAULT 0,
  created_at       INTEGER NOT NULL,
  updated_at       INTEGER NOT NULL,
  merged_into      TEXT
);

CREATE TABLE IF NOT EXISTS item_concepts (
  item_id      TEXT NOT NULL,
  concept_id   TEXT NOT NULL,
  confidence   REAL NOT NULL,
  source       TEXT NOT NULL,
  created_at   INTEGER NOT NULL,
  PRIMARY KEY (item_id, concept_id)
);

CREATE TABLE IF NOT EXISTS relations (
  id              TEXT PRIMARY KEY,
  from_kind       TEXT NOT NULL,
  from_id         TEXT NOT NULL,
  to_kind         TEXT NOT NULL,
  to_id           TEXT NOT NULL,
  relation_type   TEXT NOT NULL,
  confidence      REAL NOT NULL,
  source          TEXT NOT NULL,
  rationale       TEXT,
  created_at      INTEGER NOT NULL,
  invalidated_at  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_kind, to_id);
CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_kind, from_id);

CREATE TABLE IF NOT EXISTS consistency_issues (
  id             TEXT PRIMARY KEY,
  trigger_event  TEXT NOT NULL,
  trigger_ref    TEXT,
  issue_type     TEXT NOT NULL,
  subject_ids    TEXT NOT NULL,
  description    TEXT NOT NULL,
  severity       TEXT NOT NULL,
  status         TEXT NOT NULL DEFAULT 'open',
  resolution     TEXT,
  created_at     INTEGER NOT NULL,
  resolved_at    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_issues_status ON consistency_issues(status, created_at);

CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

_SCHEMA_VERSION = "1"


class MemoryIndex:
    """SQLite sidecar index for MEMORY.md: items, concepts, relations, issues."""

    def __init__(self, db_path: Path, embedding_dim: int):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path))
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._init_schema(embedding_dim)

    def _init_schema(self, embedding_dim: int) -> None:
        self._db.executescript(_SCHEMA_SQL)
        stored_dim = self.get_meta("embedding_dim")
        if stored_dim is None:
            self.set_meta("embedding_dim", str(embedding_dim))
        elif int(stored_dim) != embedding_dim:
            raise ValueError(
                f"embedding dim mismatch: stored={stored_dim}, requested={embedding_dim}; "
                f"delete {self._db_path} to rebuild"
            )
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", _SCHEMA_VERSION)
        self._db.commit()

    def list_tables(self) -> list[str]:
        cur = self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [r["name"] for r in cur.fetchall()]

    def get_meta(self, key: str) -> str | None:
        cur = self._db.execute("SELECT value FROM metadata WHERE key=?", (key,))
        row = cur.fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO metadata(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
