"""MemoryIndex — SQLite sidecar index for MEMORY.md."""

from __future__ import annotations

import math
import sqlite3
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ItemRecord:
    id: str
    source_file: str
    section_path: str
    item_type: str
    content: str
    content_hash: str
    embedding: bytes | None
    created_at: int
    updated_at: int
    removed_at: int | None


def _new_id() -> str:
    return uuid.uuid4().hex


def _row_to_item(row: sqlite3.Row) -> ItemRecord:
    return ItemRecord(
        id=row["id"],
        source_file=row["source_file"],
        section_path=row["section_path"],
        item_type=row["item_type"],
        content=row["content"],
        content_hash=row["content_hash"],
        embedding=row["embedding"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        removed_at=row["removed_at"],
    )


def _unpack_floats(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


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

    # ---- items ----
    def upsert_item(self, item: ItemRecord) -> str:
        if not item.id:
            item.id = _new_id()
            self._db.execute(
                "INSERT INTO memory_items "
                "(id, source_file, section_path, item_type, content, content_hash, "
                " embedding, created_at, updated_at, removed_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (item.id, item.source_file, item.section_path, item.item_type,
                 item.content, item.content_hash, item.embedding,
                 item.created_at, item.updated_at, item.removed_at),
            )
        else:
            self._db.execute(
                "UPDATE memory_items SET source_file=?, section_path=?, item_type=?, "
                "content=?, content_hash=?, embedding=?, updated_at=?, removed_at=? "
                "WHERE id=?",
                (item.source_file, item.section_path, item.item_type,
                 item.content, item.content_hash, item.embedding,
                 item.updated_at, item.removed_at, item.id),
            )
        self._db.commit()
        return item.id

    def tombstone_item(self, item_id: str) -> None:
        now = int(time.time())
        self._db.execute(
            "UPDATE memory_items SET removed_at=?, updated_at=? WHERE id=? AND removed_at IS NULL",
            (now, now, item_id),
        )
        self._db.commit()

    def get_item(self, item_id: str) -> ItemRecord | None:
        cur = self._db.execute("SELECT * FROM memory_items WHERE id=?", (item_id,))
        row = cur.fetchone()
        return _row_to_item(row) if row else None

    def list_items(
        self,
        source_file: str | None = None,
        alive: bool = True,
    ) -> list[ItemRecord]:
        clauses: list[str] = []
        params: list = []
        if source_file is not None:
            clauses.append("source_file = ?")
            params.append(source_file)
        if alive:
            clauses.append("removed_at IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = self._db.execute(f"SELECT * FROM memory_items {where} ORDER BY created_at", params)
        return [_row_to_item(r) for r in cur.fetchall()]

    def find_similar_items(
        self,
        embedding: bytes,
        top_k: int = 5,
        threshold: float = 0.92,
    ) -> list[tuple[str, float]]:
        """Linear-scan cosine similarity over alive items with embeddings."""
        query_vec = _unpack_floats(embedding)
        cur = self._db.execute(
            "SELECT id, embedding FROM memory_items "
            "WHERE removed_at IS NULL AND embedding IS NOT NULL"
        )
        hits: list[tuple[str, float]] = []
        for row in cur.fetchall():
            other = _unpack_floats(row["embedding"])
            if len(other) != len(query_vec):
                continue
            score = _cosine(query_vec, other)
            if score >= threshold:
                hits.append((row["id"], score))
        hits.sort(key=lambda x: -x[1])
        return hits[:top_k]

    def close(self) -> None:
        self._db.close()
