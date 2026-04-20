"""Tests for the restructured MemoryStore — pure file I/O layer."""

from datetime import datetime
import json
from pathlib import Path

import pytest

from nanobot.agent.memory import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


class TestMemoryStoreBasicIO:
    def test_read_memory_returns_empty_when_missing(self, store):
        assert store.read_memory() == ""

    def test_write_and_read_memory(self, store):
        store.write_memory("hello")
        assert store.read_memory() == "hello"

    def test_read_soul_returns_empty_when_missing(self, store):
        assert store.read_soul() == ""

    def test_write_and_read_soul(self, store):
        store.write_soul("soul content")
        assert store.read_soul() == "soul content"

    def test_read_user_returns_empty_when_missing(self, store):
        assert store.read_user() == ""

    def test_write_and_read_user(self, store):
        store.write_user("user content")
        assert store.read_user() == "user content"

    def test_get_memory_context_returns_empty_when_missing(self, store):
        assert store.get_memory_context() == ""

    def test_get_memory_context_returns_formatted_content(self, store):
        store.write_memory("important fact")
        ctx = store.get_memory_context()
        assert "Long-term Memory" in ctx
        assert "important fact" in ctx


class TestHistoryWithCursor:
    def test_append_history_returns_cursor(self, store):
        cursor = store.append_history("event 1")
        assert cursor == 1
        cursor2 = store.append_history("event 2")
        assert cursor2 == 2

    def test_append_history_stores_cursor_in_db(self, store):
        store.append_history("event 1")
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert entries[0]["cursor"] == 1

    def test_cursor_persists_across_appends(self, store):
        store.append_history("event 1")
        store.append_history("event 2")
        cursor = store.append_history("event 3")
        assert cursor == 3

    def test_read_unprocessed_history(self, store):
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        entries = store.read_unprocessed_history(since_cursor=1)
        assert len(entries) == 2
        assert entries[0]["cursor"] == 2

    def test_read_unprocessed_history_returns_all_when_cursor_zero(self, store):
        store.append_history("event 1")
        store.append_history("event 2")
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2

    def test_read_unprocessed_skips_entries_without_cursor(self, store):
        """Regression: entries missing the cursor key should be silently skipped."""
        store.history_file.write_text(
            '{"timestamp": "2026-04-01 10:00", "content": "no cursor"}\n'
            '{"cursor": 2, "timestamp": "2026-04-01 10:01", "content": "valid"}\n'
            '{"cursor": 3, "timestamp": "2026-04-01 10:02", "content": "also valid"}\n',
            encoding="utf-8",
        )
        entries = store.read_unprocessed_history(since_cursor=0)
        assert [e["cursor"] for e in entries] == [2, 3]

    def test_next_cursor_falls_back_when_last_entry_has_no_cursor(self, store):
        """Regression: _next_cursor should not KeyError on entries without cursor."""
        store.history_file.write_text(
            '{"timestamp": "2026-04-01 10:01", "content": "no cursor"}\n',
            encoding="utf-8",
        )
        # Delete .cursor file so _next_cursor falls back to reading JSONL
        store._cursor_file.unlink(missing_ok=True)
        # Last entry has no cursor — should safely return 1, not KeyError
        cursor = store.append_history("new event")
        assert cursor == 1

    def test_compact_history_drops_oldest(self, tmp_path):
        store = MemoryStore(tmp_path, max_history_entries=2)
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        store.append_history("event 4")
        store.append_history("event 5")
        store.compact_history()
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2
        assert entries[0]["cursor"] in {4, 5}


class TestDreamCursor:
    def test_initial_cursor_is_zero(self, store):
        assert store.get_last_dream_cursor() == 0

    def test_set_and_get_cursor(self, store):
        store.set_last_dream_cursor(5)
        assert store.get_last_dream_cursor() == 5

    def test_cursor_persists(self, store):
        store.set_last_dream_cursor(3)
        store2 = MemoryStore(store.workspace)
        assert store2.get_last_dream_cursor() == 3


class TestLegacyHistoryMigration:
    def test_read_unprocessed_history_handles_entries_via_append(self, store):
        """Entries appended via append_history() are correctly returned."""
        store.append_history("Old event")
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert entries[0]["cursor"] == 1

    def test_migrates_legacy_history_md_preserving_partial_entries(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        legacy_file = memory_dir / "HISTORY.md"
        legacy_content = (
            "[2026-04-01 10:00] User prefers dark mode.\n\n"
            "[2026-04-01 10:05] [RAW] 2 messages\n"
            "[2026-04-01 10:04] USER: hello\n"
            "[2026-04-01 10:04] ASSISTANT: hi\n\n"
            "Legacy chunk without timestamp.\n"
            "Keep whatever content we can recover.\n"
        )
        legacy_file.write_text(legacy_content, encoding="utf-8")

        store = MemoryStore(tmp_path)
        fallback_timestamp = datetime.fromtimestamp(
            (memory_dir / "HISTORY.md.bak").stat().st_mtime,
        ).strftime("%Y-%m-%d %H:%M")

        entries = store.read_unprocessed_history(since_cursor=0)
        assert [entry["cursor"] for entry in entries] == [1, 2, 3]
        assert entries[0]["timestamp"] == "2026-04-01 10:00"
        assert entries[0]["content"] == "User prefers dark mode."
        assert entries[1]["timestamp"] == "2026-04-01 10:05"
        assert entries[1]["content"].startswith("[RAW] 2 messages")
        assert "USER: hello" in entries[1]["content"]
        assert entries[2]["timestamp"] == fallback_timestamp
        assert entries[2]["content"].startswith("Legacy chunk without timestamp.")
        # Migration goes directly to SQLite — no .cursor or .dream_cursor files
        assert not store._cursor_file.exists()
        assert not store._dream_cursor_file.exists()
        assert not legacy_file.exists()
        assert (memory_dir / "HISTORY.md.bak").read_text(encoding="utf-8") == legacy_content
        # Dream cursor is set in SQLite metadata
        assert store.get_last_dream_cursor() == 3

    def test_migrates_consecutive_entries_without_blank_lines(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        legacy_file = memory_dir / "HISTORY.md"
        legacy_content = (
            "[2026-04-01 10:00] First event.\n"
            "[2026-04-01 10:01] Second event.\n"
            "[2026-04-01 10:02] Third event.\n"
        )
        legacy_file.write_text(legacy_content, encoding="utf-8")

        store = MemoryStore(tmp_path)

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 3
        assert [entry["content"] for entry in entries] == [
            "First event.",
            "Second event.",
            "Third event.",
        ]

    def test_raw_archive_stays_single_entry_while_following_events_split(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        legacy_file = memory_dir / "HISTORY.md"
        legacy_content = (
            "[2026-04-01 10:05] [RAW] 2 messages\n"
            "[2026-04-01 10:04] USER: hello\n"
            "[2026-04-01 10:04] ASSISTANT: hi\n"
            "[2026-04-01 10:06] Normal event after raw block.\n"
        )
        legacy_file.write_text(legacy_content, encoding="utf-8")

        store = MemoryStore(tmp_path)

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2
        assert entries[0]["content"].startswith("[RAW] 2 messages")
        assert "USER: hello" in entries[0]["content"]
        assert entries[1]["content"] == "Normal event after raw block."

    def test_nonstandard_date_headers_still_start_new_entries(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        legacy_file = memory_dir / "HISTORY.md"
        legacy_content = (
            "[2026-03-25\u20132026-04-02] Multi-day summary.\n"
            "[2026-03-26/27] Cross-day summary.\n"
        )
        legacy_file.write_text(legacy_content, encoding="utf-8")

        store = MemoryStore(tmp_path)
        fallback_timestamp = datetime.fromtimestamp(
            (memory_dir / "HISTORY.md.bak").stat().st_mtime,
        ).strftime("%Y-%m-%d %H:%M")

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2
        assert entries[0]["timestamp"] == fallback_timestamp
        assert entries[0]["content"] == "[2026-03-25\u20132026-04-02] Multi-day summary."
        assert entries[1]["timestamp"] == fallback_timestamp
        assert entries[1]["content"] == "[2026-03-26/27] Cross-day summary."

    def test_existing_history_db_skips_legacy_migration(self, tmp_path):
        """Existing history.db prevents legacy HISTORY.md migration."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        # Create a pre-existing history.db with data
        pre_store = MemoryStore(tmp_path)
        pre_store.append_history("existing")
        # Now create a legacy file that should NOT be migrated
        legacy_file = memory_dir / "HISTORY.md"
        legacy_file.write_text("[2026-04-01 10:00] legacy\n\n", encoding="utf-8")

        store = MemoryStore(tmp_path)

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert entries[0]["content"] == "existing"
        assert legacy_file.exists()
        assert not (memory_dir / "HISTORY.md.bak").exists()

    def test_empty_history_db_still_allows_legacy_migration(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        legacy_file = memory_dir / "HISTORY.md"
        legacy_file.write_text("[2026-04-01 10:00] legacy\n\n", encoding="utf-8")

        store = MemoryStore(tmp_path)

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert entries[0]["cursor"] == 1
        assert entries[0]["timestamp"] == "2026-04-01 10:00"
        assert entries[0]["content"] == "legacy"
        assert not legacy_file.exists()
        assert (memory_dir / "HISTORY.md.bak").exists()

    def test_migrates_legacy_history_with_invalid_utf8_bytes(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        legacy_file = memory_dir / "HISTORY.md"
        legacy_file.write_bytes(
            b"[2026-04-01 10:00] Broken \xff data still needs migration.\n\n"
        )

        store = MemoryStore(tmp_path)

        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 1
        assert entries[0]["timestamp"] == "2026-04-01 10:00"
        assert "Broken" in entries[0]["content"]
        assert "migration." in entries[0]["content"]


class TestSQLiteBackend:
    """Tests for SQLite-backed history storage."""

    def test_append_history_returns_cursor(self, store):
        cursor = store.append_history("event 1")
        assert cursor == 1
        cursor2 = store.append_history("event 2")
        assert cursor2 == 2

    def test_history_db_file_created(self, store):
        store.append_history("event 1")
        assert (store.memory_dir / "history.db").exists()

    def test_read_unprocessed_history(self, store):
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        entries = store.read_unprocessed_history(since_cursor=1)
        assert len(entries) == 2
        assert entries[0]["cursor"] == 2

    def test_compact_history_drops_oldest(self, tmp_path):
        s = MemoryStore(tmp_path, max_history_entries=2)
        for i in range(5):
            s.append_history(f"event {i+1}")
        s.compact_history()
        entries = s.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2
        assert entries[0]["cursor"] == 4

    def test_dream_cursor_via_metadata_table(self, store):
        assert store.get_last_dream_cursor() == 0
        store.set_last_dream_cursor(5)
        assert store.get_last_dream_cursor() == 5

    def test_dream_cursor_persists_across_instances(self, store):
        store.set_last_dream_cursor(3)
        store2 = MemoryStore(store.workspace)
        assert store2.get_last_dream_cursor() == 3

    def test_no_jsonl_file_created(self, store):
        """After SQLite upgrade, history.jsonl should NOT be created for new workspaces."""
        store.append_history("event 1")
        assert not store.history_file.exists()


class TestFTS5Search:
    """Tests for full-text search via FTS5."""

    def test_search_history_basic(self, store):
        store.append_history("用户请求编写自动部署脚本")
        store.append_history("修复登录页面的 CSS 样式问题")
        store.append_history("讨论部署流水线的优化方案")
        results = store.search_history("部署")
        assert len(results) >= 2
        contents = [r["content"] for r in results]
        assert any("部署脚本" in c for c in contents)
        assert any("部署流水线" in c for c in contents)

    def test_search_history_english(self, store):
        store.append_history("Fixed the authentication bug in login flow")
        store.append_history("Added new CSS styles for dashboard")
        results = store.search_history("authentication")
        assert len(results) == 1
        assert "authentication" in results[0]["content"]

    def test_search_history_returns_ranked(self, store):
        store.append_history("部署脚本第一版")
        store.append_history("完全无关的内容")
        results = store.search_history("部署")
        assert len(results) == 1
        assert "rank" in results[0]

    def test_search_history_respects_limit(self, store):
        for i in range(10):
            store.append_history(f"测试条目 {i}")
        results = store.search_history("测试", limit=3)
        assert len(results) == 3

    def test_search_history_empty_result(self, store):
        store.append_history("some content")
        results = store.search_history("不存在的关键词xyz")
        assert results == []

    def test_search_history_empty_db(self, store):
        results = store.search_history("anything")
        assert results == []


class TestQueryHistory:
    """Tests for time-range queries."""

    def test_query_history_all(self, store):
        store.append_history("event 1")
        store.append_history("event 2")
        results = store.query_history()
        assert len(results) == 2

    def test_query_history_with_limit(self, store):
        for i in range(10):
            store.append_history(f"event {i}")
        results = store.query_history(limit=3)
        assert len(results) == 3


class TestJSONLToSQLiteMigration:
    """Tests for automatic JSONL → SQLite migration."""

    def test_migrates_existing_jsonl(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        jsonl = memory_dir / "history.jsonl"
        jsonl.write_text(
            '{"cursor": 1, "timestamp": "2026-04-01 10:00", "content": "event one"}\n'
            '{"cursor": 2, "timestamp": "2026-04-01 11:00", "content": "event two"}\n',
            encoding="utf-8",
        )
        cursor_file = memory_dir / ".cursor"
        cursor_file.write_text("2", encoding="utf-8")
        dream_cursor = memory_dir / ".dream_cursor"
        dream_cursor.write_text("1", encoding="utf-8")

        store = MemoryStore(tmp_path)
        entries = store.read_unprocessed_history(since_cursor=0)
        assert len(entries) == 2
        assert entries[0]["content"] == "event one"
        assert entries[1]["content"] == "event two"
        assert store.get_last_dream_cursor() == 1
        assert not jsonl.exists()
        assert (memory_dir / "history.jsonl.bak").exists()
        assert not cursor_file.exists()
        assert not dream_cursor.exists()
        assert (memory_dir / "history.db").exists()

    def test_migration_preserves_original_cursors(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        jsonl = memory_dir / "history.jsonl"
        jsonl.write_text(
            '{"cursor": 5, "timestamp": "2026-04-01 10:00", "content": "late start"}\n'
            '{"cursor": 6, "timestamp": "2026-04-01 11:00", "content": "next"}\n',
            encoding="utf-8",
        )
        (memory_dir / ".cursor").write_text("6", encoding="utf-8")

        store = MemoryStore(tmp_path)
        entries = store.read_unprocessed_history(since_cursor=4)
        assert len(entries) == 2
        assert entries[0]["cursor"] == 5

    def test_no_jsonl_means_no_migration(self, tmp_path):
        store = MemoryStore(tmp_path)
        store.append_history("fresh start")
        assert not (tmp_path / "memory" / "history.jsonl.bak").exists()

    def test_migrated_entries_are_searchable(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        jsonl = memory_dir / "history.jsonl"
        jsonl.write_text(
            '{"cursor": 1, "timestamp": "2026-04-01 10:00", "content": "部署脚本讨论"}\n',
            encoding="utf-8",
        )
        (memory_dir / ".cursor").write_text("1", encoding="utf-8")

        store = MemoryStore(tmp_path)
        results = store.search_history("部署")
        assert len(results) == 1


class TestMemoryVecTable:
    def test_vec_available_when_dimensions_set(self, tmp_path):
        from nanobot.agent.memory import MemoryStore
        store = MemoryStore(tmp_path, embedding_dimensions=4)
        assert store.vec_available is True

    def test_vec_available_false_when_no_dimensions(self, tmp_path):
        from nanobot.agent.memory import MemoryStore
        store = MemoryStore(tmp_path)
        assert store.vec_available is False

    def test_vec_table_created_with_configured_dimensions(self, tmp_path):
        from nanobot.agent.memory import MemoryStore
        store = MemoryStore(tmp_path, embedding_dimensions=4)
        # Force DB init
        store.append_history("seed")
        db = store._get_db()
        row = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='history_vec'"
        ).fetchone()
        assert row is not None

    def test_vec_table_rebuilt_when_dimensions_change(self, tmp_path):
        from nanobot.agent.memory import MemoryStore
        store1 = MemoryStore(tmp_path, embedding_dimensions=4)
        store1.append_history("seed")
        # Close before re-opening with different dim
        store1._db.close()
        store1._db = None

        store2 = MemoryStore(tmp_path, embedding_dimensions=8)
        store2.append_history("seed2")
        db = store2._get_db()
        dim_row = db.execute(
            "SELECT value FROM metadata WHERE key='embedding_dimensions'"
        ).fetchone()
        assert dim_row[0] == "8"
