"""Tests for MemoryIndex — SQLite sidecar index for MEMORY.md."""

from pathlib import Path

import pytest

from nanobot.agent.memory_index import MemoryIndex


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
