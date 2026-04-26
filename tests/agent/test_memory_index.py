"""Tests for MemoryIndex — SQLite sidecar index for MEMORY.md."""

import time
from pathlib import Path

import pytest

from nanobot.agent.memory_index import ItemRecord, MemoryIndex


@pytest.fixture
def index(tmp_path: Path) -> MemoryIndex:
    return MemoryIndex(db_path=tmp_path / "index.db", embedding_dim=1536)


class TestSchemaCreation:
    def test_tables_exist(self, index: MemoryIndex):
        tables = index.list_tables()
        assert {"memory_items", "concepts", "item_concepts", "relations",
                "consistency_issues", "metadata"} <= set(tables)

    def test_metadata_defaults(self, index: MemoryIndex):
        assert index.get_meta("schema_version") == "1"
        assert index.get_meta("embedding_dim") == "1536"
        assert index.get_meta("last_reconciled_commit") is None

    def test_set_and_get_meta(self, index: MemoryIndex):
        index.set_meta("last_reconciled_commit", "abc123")
        assert index.get_meta("last_reconciled_commit") == "abc123"


class TestItemsCRUD:
    def _make_item(self, content: str = "foo", section: str = "root") -> ItemRecord:
        now = int(time.time())
        return ItemRecord(
            id="",  # filled by upsert
            source_file="memory/MEMORY.md",
            section_path=section,
            item_type="list_item",
            content=content,
            content_hash=f"hash-{content}",
            embedding=None,
            created_at=now,
            updated_at=now,
            removed_at=None,
        )

    def test_upsert_assigns_id(self, index: MemoryIndex):
        item = self._make_item()
        item_id = index.upsert_item(item)
        assert item_id
        assert index.get_item(item_id).content == "foo"

    def test_list_items_alive_only_by_default(self, index: MemoryIndex):
        a = index.upsert_item(self._make_item("a"))
        b = index.upsert_item(self._make_item("b"))
        index.tombstone_item(b)
        ids = [it.id for it in index.list_items()]
        assert a in ids and b not in ids

    def test_list_items_include_tombstoned(self, index: MemoryIndex):
        a = index.upsert_item(self._make_item("a"))
        index.tombstone_item(a)
        ids = [it.id for it in index.list_items(alive=False)]
        assert a in ids

    def test_upsert_existing_id_updates_content(self, index: MemoryIndex):
        item = self._make_item("v1")
        item_id = index.upsert_item(item)
        item.id = item_id
        item.content = "v2"
        item.content_hash = "hash-v2"
        index.upsert_item(item)
        assert index.get_item(item_id).content == "v2"

    def test_find_similar_items_returns_hits_above_threshold(self, index: MemoryIndex):
        import struct
        def vec(vals): return struct.pack(f"{len(vals)}f", *vals)
        # fixture uses embedding_dim=1536; use 3 floats padded with zeros
        def pad(vals):
            return vec(list(vals) + [0.0] * (1536 - len(vals)))
        a = self._make_item("a"); a.embedding = pad([1.0, 0.0, 0.0])
        b = self._make_item("b"); b.embedding = pad([0.99, 0.1, 0.0])
        c = self._make_item("c"); c.embedding = pad([0.0, 1.0, 0.0])
        index.upsert_item(a); index.upsert_item(b); index.upsert_item(c)
        results = index.find_similar_items(pad([1.0, 0.0, 0.0]), top_k=2, threshold=0.9)
        assert len(results) == 2
        # highest first
        assert results[0][1] >= results[1][1]
