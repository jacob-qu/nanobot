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


@dataclass
class Concept:
    id: str
    name: str
    description: str | None
    centroid_embed: bytes | None
    member_count: int
    created_at: int
    updated_at: int
    merged_into: str | None


def _row_to_concept(row: sqlite3.Row) -> Concept:
    return Concept(
        id=row["id"], name=row["name"], description=row["description"],
        centroid_embed=row["centroid_embed"], member_count=row["member_count"],
        created_at=row["created_at"], updated_at=row["updated_at"],
        merged_into=row["merged_into"],
    )


@dataclass
class Relation:
    id: str
    from_kind: str
    from_id: str
    to_kind: str
    to_id: str
    relation_type: str
    confidence: float
    source: str
    rationale: str | None
    created_at: int
    invalidated_at: int | None


@dataclass
class ConsistencyIssue:
    id: str
    trigger_event: str
    trigger_ref: str | None
    issue_type: str
    subject_ids: str  # JSON
    description: str
    severity: str
    status: str
    resolution: str | None
    created_at: int
    resolved_at: int | None


@dataclass
class ImpactResult:
    target_kind: str
    target_id: str
    incoming_edges: list[Relation]
    outgoing_edges: list[Relation]
    transitive_nodes: list[tuple[str, str, int]]  # (kind, id, depth)


def _row_to_relation(row: sqlite3.Row) -> Relation:
    return Relation(
        id=row["id"], from_kind=row["from_kind"], from_id=row["from_id"],
        to_kind=row["to_kind"], to_id=row["to_id"],
        relation_type=row["relation_type"], confidence=row["confidence"],
        source=row["source"], rationale=row["rationale"],
        created_at=row["created_at"], invalidated_at=row["invalidated_at"],
    )


def _row_to_issue(row: sqlite3.Row) -> ConsistencyIssue:
    return ConsistencyIssue(
        id=row["id"], trigger_event=row["trigger_event"],
        trigger_ref=row["trigger_ref"], issue_type=row["issue_type"],
        subject_ids=row["subject_ids"], description=row["description"],
        severity=row["severity"], status=row["status"],
        resolution=row["resolution"], created_at=row["created_at"],
        resolved_at=row["resolved_at"],
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

    # ---- concepts ----
    def upsert_concept(self, c: Concept) -> str:
        if not c.id:
            c.id = _new_id()
            self._db.execute(
                "INSERT INTO concepts (id, name, description, centroid_embed, "
                "member_count, created_at, updated_at, merged_into) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (c.id, c.name, c.description, c.centroid_embed, c.member_count,
                 c.created_at, c.updated_at, c.merged_into),
            )
        else:
            self._db.execute(
                "UPDATE concepts SET name=?, description=?, centroid_embed=?, "
                "member_count=?, updated_at=?, merged_into=? WHERE id=?",
                (c.name, c.description, c.centroid_embed, c.member_count,
                 c.updated_at, c.merged_into, c.id),
            )
        self._db.commit()
        return c.id

    def get_concept(self, concept_id: str) -> Concept | None:
        cur = self._db.execute("SELECT * FROM concepts WHERE id=?", (concept_id,))
        row = cur.fetchone()
        return _row_to_concept(row) if row else None

    def find_concept_by_name(self, name: str, fuzzy: bool = True) -> list[Concept]:
        if fuzzy:
            pattern = f"%{name}%"
            cur = self._db.execute(
                "SELECT * FROM concepts WHERE merged_into IS NULL AND name LIKE ? "
                "ORDER BY created_at",
                (pattern,),
            )
        else:
            cur = self._db.execute(
                "SELECT * FROM concepts WHERE merged_into IS NULL AND name=? "
                "ORDER BY created_at",
                (name,),
            )
        return [_row_to_concept(r) for r in cur.fetchall()]

    def merge_concept(self, src_id: str, dst_id: str) -> None:
        now = int(time.time())
        self._db.execute(
            "UPDATE concepts SET merged_into=?, updated_at=? WHERE id=?",
            (dst_id, now, src_id),
        )
        # repoint item_concepts
        self._db.execute(
            "UPDATE OR IGNORE item_concepts SET concept_id=? WHERE concept_id=?",
            (dst_id, src_id),
        )
        self._db.execute("DELETE FROM item_concepts WHERE concept_id=?", (src_id,))
        self._db.commit()

    def dedup_concepts_by_name(self) -> int:
        """Merge concepts with identical (normalized) names into the oldest.

        Returns the number of concepts merged away.
        """
        cur = self._db.execute(
            "SELECT id, name, created_at FROM concepts "
            "WHERE merged_into IS NULL ORDER BY created_at"
        )
        groups: dict[str, list[tuple[str, int]]] = {}
        for row in cur.fetchall():
            key = row["name"].strip()
            groups.setdefault(key, []).append((row["id"], row["created_at"]))

        merged = 0
        for name, rows in groups.items():
            if len(rows) < 2:
                continue
            # Oldest wins (first in ORDER BY created_at)
            keeper_id = rows[0][0]
            for src_id, _ in rows[1:]:
                self.merge_concept(src_id, keeper_id)
                merged += 1
        # Refresh member_count on survivors
        self._db.execute(
            "UPDATE concepts SET member_count = ("
            "  SELECT COUNT(*) FROM item_concepts WHERE concept_id = concepts.id"
            ") WHERE merged_into IS NULL"
        )
        self._db.commit()
        return merged

    def find_similar_concepts(
        self, embedding: bytes, top_k: int = 5,
    ) -> list[tuple[str, float]]:
        qv = _unpack_floats(embedding)
        cur = self._db.execute(
            "SELECT id, centroid_embed FROM concepts "
            "WHERE merged_into IS NULL AND centroid_embed IS NOT NULL"
        )
        hits: list[tuple[str, float]] = []
        for r in cur.fetchall():
            other = _unpack_floats(r["centroid_embed"])
            if len(other) != len(qv):
                continue
            hits.append((r["id"], _cosine(qv, other)))
        hits.sort(key=lambda x: -x[1])
        return hits[:top_k]

    # ---- item_concepts ----
    def link_item_concept(
        self, item_id: str, concept_id: str, confidence: float, source: str,
    ) -> None:
        now = int(time.time())
        self._db.execute(
            "INSERT OR REPLACE INTO item_concepts "
            "(item_id, concept_id, confidence, source, created_at) "
            "VALUES (?,?,?,?,?)",
            (item_id, concept_id, confidence, source, now),
        )
        self._db.commit()

    def unlink_item_concept(self, item_id: str, concept_id: str) -> None:
        self._db.execute(
            "DELETE FROM item_concepts WHERE item_id=? AND concept_id=?",
            (item_id, concept_id),
        )
        self._db.commit()

    def list_concepts_for_item(self, item_id: str) -> list[str]:
        cur = self._db.execute(
            "SELECT concept_id FROM item_concepts WHERE item_id=?", (item_id,),
        )
        return [r["concept_id"] for r in cur.fetchall()]

    def list_items_for_concept(self, concept_id: str) -> list[str]:
        cur = self._db.execute(
            "SELECT item_id FROM item_concepts WHERE concept_id=?", (concept_id,),
        )
        return [r["item_id"] for r in cur.fetchall()]

    # ---- relations ----
    def add_relation(self, r: Relation) -> str:
        if not r.id:
            r.id = _new_id()
        self._db.execute(
            "INSERT INTO relations (id, from_kind, from_id, to_kind, to_id, "
            "relation_type, confidence, source, rationale, created_at, invalidated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (r.id, r.from_kind, r.from_id, r.to_kind, r.to_id, r.relation_type,
             r.confidence, r.source, r.rationale, r.created_at, r.invalidated_at),
        )
        self._db.commit()
        return r.id

    def invalidate_relation(self, relation_id: str) -> None:
        now = int(time.time())
        self._db.execute(
            "UPDATE relations SET invalidated_at=? WHERE id=? AND invalidated_at IS NULL",
            (now, relation_id),
        )
        self._db.commit()

    def relations_from(
        self, kind: str, id: str, types: list[str] | None = None,
    ) -> list[Relation]:
        sql = ("SELECT * FROM relations WHERE from_kind=? AND from_id=? "
               "AND invalidated_at IS NULL")
        params: list = [kind, id]
        if types:
            sql += f" AND relation_type IN ({','.join('?' * len(types))})"
            params.extend(types)
        cur = self._db.execute(sql, params)
        return [_row_to_relation(r) for r in cur.fetchall()]

    def relations_to(
        self, kind: str, id: str, types: list[str] | None = None,
    ) -> list[Relation]:
        sql = ("SELECT * FROM relations WHERE to_kind=? AND to_id=? "
               "AND invalidated_at IS NULL")
        params: list = [kind, id]
        if types:
            sql += f" AND relation_type IN ({','.join('?' * len(types))})"
            params.extend(types)
        cur = self._db.execute(sql, params)
        return [_row_to_relation(r) for r in cur.fetchall()]

    # ---- impact ----
    def query_impact(self, kind: str, id: str, depth: int = 2) -> ImpactResult:
        incoming = self.relations_to(kind, id)
        outgoing = self.relations_from(kind, id)
        visited: set[tuple[str, str]] = {(kind, id)}
        transitive: list[tuple[str, str, int]] = []
        frontier: list[tuple[str, str, int]] = [
            (e.from_kind, e.from_id, 1) for e in incoming
        ]
        while frontier:
            nxt: list[tuple[str, str, int]] = []
            for fk, fid, d in frontier:
                if (fk, fid) in visited:
                    continue
                visited.add((fk, fid))
                transitive.append((fk, fid, d))
                if d < depth:
                    for e in self.relations_to(fk, fid):
                        nxt.append((e.from_kind, e.from_id, d + 1))
            frontier = nxt
        return ImpactResult(
            target_kind=kind, target_id=id,
            incoming_edges=incoming, outgoing_edges=outgoing,
            transitive_nodes=transitive,
        )

    # ---- issues ----
    def add_issue(self, issue: ConsistencyIssue) -> str:
        if not issue.id:
            issue.id = _new_id()
        self._db.execute(
            "INSERT INTO consistency_issues (id, trigger_event, trigger_ref, "
            "issue_type, subject_ids, description, severity, status, resolution, "
            "created_at, resolved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (issue.id, issue.trigger_event, issue.trigger_ref, issue.issue_type,
             issue.subject_ids, issue.description, issue.severity, issue.status,
             issue.resolution, issue.created_at, issue.resolved_at),
        )
        self._db.commit()
        return issue.id

    def list_open_issues(
        self, severity_min: str = "low",
    ) -> list[ConsistencyIssue]:
        # severity order: low < medium < high
        order = {"low": 0, "medium": 1, "high": 2}
        min_rank = order.get(severity_min, 0)
        cur = self._db.execute(
            "SELECT * FROM consistency_issues WHERE status='open' ORDER BY created_at DESC"
        )
        all_issues = [_row_to_issue(r) for r in cur.fetchall()]
        return [i for i in all_issues if order.get(i.severity, 0) >= min_rank]

    def resolve_issue(self, issue_id: str, status: str, resolution: str) -> None:
        now = int(time.time())
        self._db.execute(
            "UPDATE consistency_issues SET status=?, resolution=?, resolved_at=? "
            "WHERE id=?",
            (status, resolution, now, issue_id),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
