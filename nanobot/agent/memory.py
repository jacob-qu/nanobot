"""Memory system: pure file I/O store, lightweight Consolidator, and Dream processor."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
import weakref
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.utils.prompt_templates import render_template
from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain, strip_think

from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.utils.gitstore import GitStore

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


# ---------------------------------------------------------------------------
# MemoryStore — pure file I/O layer
# ---------------------------------------------------------------------------

class MemoryStore:
    """Pure file I/O for memory files: MEMORY.md, history (SQLite), SOUL.md, USER.md."""

    _DEFAULT_MAX_HISTORY = 1000
    _LEGACY_ENTRY_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*")
    _LEGACY_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*")
    _LEGACY_RAW_MESSAGE_RE = re.compile(
        r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+[A-Z][A-Z0-9_]*(?:\s+\[tools:\s*[^\]]+\])?:"
    )

    def __init__(self, workspace: Path, max_history_entries: int = _DEFAULT_MAX_HISTORY):
        self.workspace = workspace
        self.max_history_entries = max_history_entries
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.legacy_history_file = self.memory_dir / "HISTORY.md"
        self.soul_file = workspace / "SOUL.md"
        self.user_file = workspace / "USER.md"
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor"
        self._git = GitStore(workspace, tracked_files=[
            "SOUL.md", "USER.md", "memory/MEMORY.md",
        ])
        self._db: sqlite3.Connection | None = None
        self._maybe_migrate_legacy_history()   # HISTORY.md → SQLite
        self._maybe_migrate_jsonl_to_sqlite()  # history.jsonl → SQLite

    @property
    def git(self) -> GitStore:
        return self._git

    # -- generic helpers -----------------------------------------------------

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _maybe_migrate_legacy_history(self) -> None:
        """One-time upgrade from legacy HISTORY.md directly to SQLite."""
        if not self.legacy_history_file.exists():
            return
        # Skip if SQLite already has data
        db_path = self.memory_dir / "history.db"
        if db_path.exists() and db_path.stat().st_size > 0:
            return

        try:
            legacy_text = self.legacy_history_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            logger.exception("Failed to read legacy HISTORY.md for migration")
            return

        entries = self._parse_legacy_history(legacy_text)
        try:
            if entries:
                db = self._get_db()
                db.executemany(
                    "INSERT OR IGNORE INTO history (cursor, timestamp, content)"
                    " VALUES (?, ?, ?)",
                    [(e["cursor"], e["timestamp"], e["content"]) for e in entries],
                )
                # Populate FTS with CJK-spaced content
                db.execute("DELETE FROM history_fts")
                rows = db.execute("SELECT cursor, content FROM history").fetchall()
                for row in rows:
                    db.execute(
                        "INSERT INTO history_fts(rowid, content) VALUES (?, ?)",
                        (row[0], self._cjk_space(row[1])),
                    )
                last_cursor = entries[-1]["cursor"]
                self.set_last_dream_cursor(last_cursor)
                db.commit()

            backup_path = self._next_legacy_backup_path()
            self.legacy_history_file.replace(backup_path)
            logger.info(
                "Migrated legacy HISTORY.md to SQLite ({} entries)", len(entries),
            )
        except Exception:
            logger.exception("Failed to migrate legacy HISTORY.md")

    def _parse_legacy_history(self, text: str) -> list[dict[str, Any]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        fallback_timestamp = self._legacy_fallback_timestamp()
        entries: list[dict[str, Any]] = []
        chunks = self._split_legacy_history_chunks(normalized)

        for cursor, chunk in enumerate(chunks, start=1):
            timestamp = fallback_timestamp
            content = chunk
            match = self._LEGACY_TIMESTAMP_RE.match(chunk)
            if match:
                timestamp = match.group(1)
                remainder = chunk[match.end():].lstrip()
                if remainder:
                    content = remainder

            entries.append({
                "cursor": cursor,
                "timestamp": timestamp,
                "content": content,
            })
        return entries

    def _split_legacy_history_chunks(self, text: str) -> list[str]:
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        saw_blank_separator = False

        for line in lines:
            if saw_blank_separator and line.strip() and current:
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            if self._should_start_new_legacy_chunk(line, current):
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            current.append(line)
            saw_blank_separator = not line.strip()

        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _should_start_new_legacy_chunk(self, line: str, current: list[str]) -> bool:
        if not current:
            return False
        if not self._LEGACY_ENTRY_START_RE.match(line):
            return False
        if self._is_raw_legacy_chunk(current) and self._LEGACY_RAW_MESSAGE_RE.match(line):
            return False
        return True

    def _is_raw_legacy_chunk(self, lines: list[str]) -> bool:
        first_nonempty = next((line for line in lines if line.strip()), "")
        match = self._LEGACY_TIMESTAMP_RE.match(first_nonempty)
        if not match:
            return False
        return first_nonempty[match.end():].lstrip().startswith("[RAW]")

    def _legacy_fallback_timestamp(self) -> str:
        try:
            return datetime.fromtimestamp(
                self.legacy_history_file.stat().st_mtime,
            ).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _next_legacy_backup_path(self) -> Path:
        candidate = self.memory_dir / "HISTORY.md.bak"
        suffix = 2
        while candidate.exists():
            candidate = self.memory_dir / f"HISTORY.md.bak.{suffix}"
            suffix += 1
        return candidate

    def _maybe_migrate_jsonl_to_sqlite(self) -> None:
        """One-time migration: history.jsonl → SQLite."""
        db_path = self.memory_dir / "history.db"
        if db_path.exists() and db_path.stat().st_size > 0:
            return
        if not self.history_file.exists():
            return

        try:
            entries: list[dict[str, Any]] = []
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            if not entries:
                return

            dream_cursor = 0
            if self._dream_cursor_file.exists():
                try:
                    dream_cursor = int(
                        self._dream_cursor_file.read_text(encoding="utf-8").strip()
                    )
                except (ValueError, OSError):
                    pass

            db = self._get_db()
            db.executemany(
                "INSERT OR IGNORE INTO history (cursor, timestamp, content)"
                " VALUES (?, ?, ?)",
                [(e["cursor"], e["timestamp"], e["content"]) for e in entries],
            )
            # Rebuild FTS with CJK-spaced content
            db.execute("DELETE FROM history_fts")
            rows = db.execute("SELECT cursor, content FROM history").fetchall()
            for row in rows:
                db.execute(
                    "INSERT INTO history_fts(rowid, content) VALUES (?, ?)",
                    (row[0], self._cjk_space(row[1])),
                )
            db.execute(
                "INSERT OR REPLACE INTO metadata (key, value)"
                " VALUES ('dream_cursor', ?)",
                (str(dream_cursor),),
            )
            db.commit()

            backup = self.history_file.with_suffix(".jsonl.bak")
            self.history_file.rename(backup)
            for f in (self._cursor_file, self._dream_cursor_file):
                if f.exists():
                    f.unlink()
            logger.info(
                "Migrated history.jsonl to SQLite ({} entries, dream_cursor={})",
                len(entries), dream_cursor,
            )
        except Exception:
            logger.exception("Failed to migrate history.jsonl to SQLite — keeping original")

    # -- SQLite backend -------------------------------------------------------

    _CJK_RE = re.compile(r"([\u4e00-\u9fff\u3400-\u4dbf])")

    @staticmethod
    def _cjk_space(text: str) -> str:
        """Insert spaces around CJK characters so FTS5 unicode61 tokenizes them."""
        return MemoryStore._CJK_RE.sub(r" \1 ", text).strip()

    def _get_db(self) -> sqlite3.Connection:
        """Lazily initialize and return the SQLite connection."""
        if self._db is None:
            db_path = self.memory_dir / "history.db"
            self._db = sqlite3.connect(str(db_path), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.row_factory = sqlite3.Row
            self._init_db()
        return self._db

    def _init_db(self) -> None:
        """Create tables and triggers if they don't exist."""
        db = self._db
        assert db is not None
        db.executescript("""
            CREATE TABLE IF NOT EXISTS history (
                cursor     INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                source     TEXT    NOT NULL DEFAULT 'consolidator',
                created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now'))
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO metadata (key, value) VALUES ('dream_cursor', '0');
        """)
        try:
            db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(
                    content,
                    tokenize='unicode61 remove_diacritics 2'
                )
            """)
        except sqlite3.OperationalError:
            logger.warning("FTS5 extension not available; full-text search will be disabled")
        db.commit()

    # -- MEMORY.md (long-term facts) -----------------------------------------

    def read_memory(self) -> str:
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    # -- SOUL.md -------------------------------------------------------------

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        self.soul_file.write_text(content, encoding="utf-8")

    # -- USER.md -------------------------------------------------------------

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self.user_file.write_text(content, encoding="utf-8")

    # -- context injection (used by context.py) ------------------------------

    def get_memory_context(self) -> str:
        long_term = self.read_memory()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # -- history — append-only, SQLite-backed ---------------------------------

    def append_history(self, entry: str, source: str = "consolidator") -> int:
        """Append *entry* to history and return its auto-incrementing cursor."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = strip_think(entry.rstrip()) or entry.rstrip()
        db = self._get_db()
        cur = db.execute(
            "INSERT INTO history (timestamp, content, source) VALUES (?, ?, ?)",
            (ts, content, source),
        )
        rowid = cur.lastrowid
        db.execute(
            "INSERT INTO history_fts(rowid, content) VALUES (?, ?)",
            (rowid, self._cjk_space(content)),
        )
        db.commit()
        return rowid

    def read_unprocessed_history(self, since_cursor: int) -> list[dict[str, Any]]:
        """Return history entries with cursor > *since_cursor*."""
        db = self._get_db()
        rows = db.execute(
            "SELECT cursor, timestamp, content FROM history WHERE cursor > ? ORDER BY cursor",
            (since_cursor,),
        ).fetchall()
        return [dict(r) for r in rows]

    def compact_history(self) -> None:
        """Drop oldest entries if the table exceeds *max_history_entries*."""
        if self.max_history_entries <= 0:
            return
        db = self._get_db()
        count = db.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        if count <= self.max_history_entries:
            return
        # Identify rows to delete and remove from FTS index
        to_delete = db.execute(
            "SELECT cursor FROM history WHERE cursor < ("
            "  SELECT cursor FROM history ORDER BY cursor DESC LIMIT 1 OFFSET ?"
            ")",
            (self.max_history_entries - 1,),
        ).fetchall()
        for row in to_delete:
            db.execute("DELETE FROM history_fts WHERE rowid = ?", (row[0],))
        db.execute(
            "DELETE FROM history WHERE cursor < ("
            "  SELECT cursor FROM history ORDER BY cursor DESC LIMIT 1 OFFSET ?"
            ")",
            (self.max_history_entries - 1,),
        )
        db.commit()

    # -- dream cursor --------------------------------------------------------

    def get_last_dream_cursor(self) -> int:
        db = self._get_db()
        row = db.execute("SELECT value FROM metadata WHERE key = 'dream_cursor'").fetchone()
        if row:
            try:
                return int(row[0])
            except (ValueError, TypeError):
                pass
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        db = self._get_db()
        db.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('dream_cursor', ?)",
            (str(cursor),),
        )
        db.commit()

    def search_history(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """FTS5 full-text search over history entries."""
        limit = min(max(1, limit), 20)
        db = self._get_db()
        rows = db.execute(
            "SELECT h.cursor, h.timestamp, h.content, rank "
            "FROM history_fts fts "
            "JOIN history h ON h.cursor = fts.rowid "
            "WHERE history_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT ?",
            (self._cjk_space(query), limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def query_history(self, since: str | None = None, until: str | None = None,
                      limit: int = 50) -> list[dict[str, Any]]:
        """Query history entries by time range."""
        limit = min(max(1, limit), 200)
        clauses: list[str] = []
        params: list[Any] = []
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        db = self._get_db()
        rows = db.execute(
            f"SELECT cursor, timestamp, content FROM history{where} ORDER BY cursor DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    # -- message formatting utility ------------------------------------------

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    def raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to history.jsonl without LLM summarization."""
        self.append_history(
            f"[RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )



# ---------------------------------------------------------------------------
# Consolidator — lightweight token-budget triggered consolidation
# ---------------------------------------------------------------------------


class Consolidator:
    """Lightweight consolidation: summarizes evicted messages into history.jsonl."""

    _MAX_CONSOLIDATION_ROUNDS = 5
    _MAX_CHUNK_MESSAGES = 60  # hard cap per consolidation round
    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift
    _PRUNED_TOOL_PLACEHOLDER = "[Tool output cleared to save context]"
    _SUMMARY_FAILURE_COOLDOWN_SECONDS = 600
    _COMPACTION_PREFIX = (
        "[CONTEXT COMPACTION] Earlier turns were compacted. "
        "Summary below describes completed work. "
        "Use it and current state to continue, avoid repeating work:\n"
    )

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self.tail_protect_tokens = int(context_window_tokens * 0.20) if context_window_tokens and context_window_tokens > 0 else 0
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._previous_summary: dict[str, str | None] = {}
        self._summary_failure_cooldown: dict[str, float] = {}

    def _prune_old_tool_results(
        self,
        messages: list[dict[str, Any]],
        protect_tail_idx: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Replace large old tool results with a placeholder (cheap, no LLM).

        Args:
            messages: Session messages list (modified in-place).
            protect_tail_idx: Index from which to start protecting (messages
                at this index and beyond are not pruned).

        Returns:
            (messages, pruned_count)
        """
        pruned = 0
        for i in range(min(protect_tail_idx, len(messages))):
            msg = messages[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not content or content == self._PRUNED_TOOL_PLACEHOLDER:
                continue
            if isinstance(content, str) and len(content) > 200:
                messages[i] = {**msg, "content": self._PRUNED_TOOL_PLACEHOLDER}
                pruned += 1
        return messages, pruned

    @staticmethod
    def _align_boundary_forward(messages: list[dict[str, Any]], idx: int) -> int:
        """Push boundary forward past any tool results at the start position."""
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    @staticmethod
    def _align_boundary_backward(messages: list[dict[str, Any]], idx: int) -> int:
        """Pull boundary backward to avoid splitting a tool_call/result group.

        If idx falls inside consecutive tool results, walk back to find the
        parent assistant message and return its index.
        """
        if idx <= 0 or idx >= len(messages):
            return idx
        check = idx - 1
        while check >= 0 and messages[check].get("role") == "tool":
            check -= 1
        if (
            check >= 0
            and messages[check].get("role") == "assistant"
            and messages[check].get("tool_calls")
        ):
            idx = check
        return idx

    @staticmethod
    def _sanitize_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs after compression."""
        surviving_call_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # Remove orphaned results
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]

        # Insert stubs for orphaned calls
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: list[dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = (
                            tc.get("id", "") if isinstance(tc, dict)
                            else getattr(tc, "id", "")
                        )
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": (
                                    "[Result from earlier conversation"
                                    " — see context summary]"
                                ),
                                "tool_call_id": cid,
                            })
            messages = patched

        return messages

    def persist_summary_state(self, session: Session) -> None:
        """Persist _previous_summary to session metadata for restart recovery."""
        summary = self._previous_summary.get(session.key)
        if summary is not None:
            session.metadata["_consolidation_summary"] = summary
        else:
            session.metadata.pop("_consolidation_summary", None)

    def restore_summary_state(self, session: Session) -> None:
        """Restore _previous_summary from session metadata after restart."""
        if session.key not in self._previous_summary:
            stored = session.metadata.get("_consolidation_summary")
            if stored:
                self._previous_summary[session.key] = stored

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens.

        Respects tail_protect_tokens: will not advance past the point where
        remaining messages would fall below the tail budget.
        """
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        # Compute tail protection boundary
        tail_limit = len(session.messages)
        if self.tail_protect_tokens > 0:
            tail_tokens = 0
            for i in range(len(session.messages) - 1, start - 1, -1):
                tail_tokens += estimate_message_tokens(session.messages[i])
                if tail_tokens >= self.tail_protect_tokens:
                    tail_limit = i
                    break
            # Hard minimum: always keep at least 3 messages (only when tail protection active)
            tail_limit = min(tail_limit, len(session.messages) - 3)
            tail_limit = max(tail_limit, start + 1)

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, min(len(session.messages), tail_limit)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def _cap_consolidation_boundary(
        self,
        session: Session,
        end_idx: int,
    ) -> int | None:
        """Clamp the chunk size without breaking the user-turn boundary."""
        start = session.last_consolidated
        if end_idx - start <= self._MAX_CHUNK_MESSAGES:
            return end_idx

        capped_end = start + self._MAX_CHUNK_MESSAGES
        for idx in range(capped_end, start, -1):
            if session.messages[idx].get("role") == "user":
                return idx
        return None

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive(self, messages: list[dict], session_key: str = "") -> str | None:
        """Summarize messages via LLM and append to history.jsonl.

        Returns the summary text (with compaction prefix) on success, None if nothing to archive.
        """
        if not messages:
            return None

        # Check cooldown
        if session_key:
            cooldown_until = self._summary_failure_cooldown.get(session_key, 0.0)
            if time.monotonic() < cooldown_until:
                logger.debug("Skipping summary for {} during cooldown", session_key)
                self.store.raw_archive(messages)
                return None

        try:
            formatted = MemoryStore._format_messages(messages)
            previous = self._previous_summary.get(session_key) if session_key else None

            if previous:
                system_content = render_template(
                    "agent/consolidator_update.md",
                    content=formatted,
                    previous_summary=previous,
                    strip=True,
                )
            else:
                system_content = render_template(
                    "agent/consolidator_archive.md",
                    content=formatted,
                    strip=True,
                )

            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": formatted},
                ],
                tools=None,
                tool_choice=None,
            )
            summary = response.content or "[no summary]"

            # Store for iterative updates
            if session_key:
                self._previous_summary[session_key] = summary
                self._summary_failure_cooldown.pop(session_key, None)

            self.store.append_history(summary)
            return f"{self._COMPACTION_PREFIX}{summary}"
        except Exception:
            logger.warning("Consolidation LLM call failed, raw-dumping to history")
            if session_key:
                self._summary_failure_cooldown[session_key] = (
                    time.monotonic() + self._SUMMARY_FAILURE_COOLDOWN_SECONDS
                )
            self.store.raw_archive(messages)
            return None

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            self.restore_summary_state(session)
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            try:
                estimated, source = self.estimate_session_prompt_tokens(session)
            except Exception:
                logger.exception("Token estimation failed for {}", session.key)
                estimated, source = 0, "error"
            if estimated <= 0:
                return
            if estimated < budget:
                unconsolidated_count = len(session.messages) - session.last_consolidated
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}, msgs={}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    unconsolidated_count,
                )
                return

            # Phase 0: prune old tool results (cheap, no LLM call)
            tail_protect_idx = max(
                len(session.messages) - 3,
                int(len(session.messages) * 0.8),
            )
            _, prune_count = self._prune_old_tool_results(
                session.messages, protect_tail_idx=tail_protect_idx,
            )
            if prune_count:
                logger.info(
                    "Tool output pruning: cleared {} result(s) for {}",
                    prune_count, session.key,
                )
                self.sessions.save(session)
                try:
                    estimated, source = self.estimate_session_prompt_tokens(session)
                except Exception:
                    logger.exception("Token estimation failed for {}", session.key)
                    return
                if estimated < budget:
                    logger.debug(
                        "Tool pruning sufficient for {}: {}/{}", session.key, estimated, budget,
                    )
                    return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                end_idx = self._cap_consolidation_boundary(session, end_idx)
                if end_idx is None:
                    logger.debug(
                        "Token consolidation: no capped boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = self._align_boundary_backward(
                    session.messages, end_idx,
                )
                start_aligned = self._align_boundary_forward(
                    session.messages, session.last_consolidated,
                )
                if start_aligned >= end_idx:
                    return

                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.archive(chunk, session_key=session.key):
                    return
                session.last_consolidated = end_idx
                self.persist_summary_state(session)
                self.sessions.save(session)

                remaining = session.messages[session.last_consolidated:]
                sanitized = self._sanitize_tool_pairs(remaining)
                if len(sanitized) != len(remaining):
                    session.messages = (
                        session.messages[:session.last_consolidated] + sanitized
                    )
                    self.sessions.save(session)

                try:
                    estimated, source = self.estimate_session_prompt_tokens(session)
                except Exception:
                    logger.exception("Token estimation failed for {}", session.key)
                    estimated, source = 0, "error"
                if estimated <= 0:
                    return


# ---------------------------------------------------------------------------
# Dream — heavyweight cron-scheduled memory consolidation
# ---------------------------------------------------------------------------


class Dream:
    """Two-phase memory processor: analyze history.jsonl, then edit files via AgentRunner.

    Phase 1 produces an analysis summary (plain LLM call).
    Phase 2 delegates to AgentRunner with read_file / edit_file tools so the
    LLM can make targeted, incremental edits instead of replacing entire files.
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        max_batch_size: int = 20,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self._runner = AgentRunner(provider)
        self._tools = self._build_tools()

    # -- tool registry -------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        """Build a minimal tool registry for the Dream agent."""
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR
        from nanobot.agent.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool

        tools = ToolRegistry()
        workspace = self.store.workspace
        # Allow reading builtin skills for reference during skill creation
        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        tools.register(ReadFileTool(
            workspace=workspace,
            allowed_dir=workspace,
            extra_allowed_dirs=extra_read,
        ))
        tools.register(EditFileTool(workspace=workspace, allowed_dir=workspace))
        # write_file resolves relative paths from workspace root, but can only
        # write under skills/ so the prompt can safely use skills/<name>/SKILL.md.
        skills_dir = workspace / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        tools.register(WriteFileTool(workspace=workspace, allowed_dir=skills_dir))
        return tools

    # -- skill listing --------------------------------------------------------

    def _list_existing_skills(self) -> list[str]:
        """List existing skills as 'name — description' for dedup context."""
        import re as _re

        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        _DESC_RE = _re.compile(r"^description:\s*(.+)$", _re.MULTILINE | _re.IGNORECASE)
        entries: dict[str, str] = {}
        for base in (self.store.workspace / "skills", BUILTIN_SKILLS_DIR):
            if not base.exists():
                continue
            for d in base.iterdir():
                if not d.is_dir():
                    continue
                skill_md = d / "SKILL.md"
                if not skill_md.exists():
                    continue
                # Prefer workspace skills over builtin (same name)
                if d.name in entries and base == BUILTIN_SKILLS_DIR:
                    continue
                content = skill_md.read_text(encoding="utf-8")[:500]
                m = _DESC_RE.search(content)
                desc = m.group(1).strip() if m else "(no description)"
                entries[d.name] = desc
        return [f"{name} — {desc}" for name, desc in sorted(entries.items())]

    # -- main entry ----------------------------------------------------------

    async def run(self) -> bool:
        """Process unprocessed history entries. Returns True if work was done."""
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        last_cursor = self.store.get_last_dream_cursor()
        entries = self.store.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return False

        batch = entries[: self.max_batch_size]
        logger.info(
            "Dream: processing {} entries (cursor {}→{}), batch={}",
            len(entries), last_cursor, batch[-1]["cursor"], len(batch),
        )

        # Build history text for LLM
        history_text = "\n".join(
            f"[{e['timestamp']}] {e['content']}" for e in batch
        )

        # Current file contents
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_memory = self.store.read_memory() or "(empty)"
        current_soul = self.store.read_soul() or "(empty)"
        current_user = self.store.read_user() or "(empty)"

        file_context = (
            f"## Current Date\n{current_date}\n\n"
            f"## Current MEMORY.md ({len(current_memory)} chars)\n{current_memory}\n\n"
            f"## Current SOUL.md ({len(current_soul)} chars)\n{current_soul}\n\n"
            f"## Current USER.md ({len(current_user)} chars)\n{current_user}"
        )

        # Phase 1: Analyze (no skills list — dedup is Phase 2's job)
        phase1_prompt = (
            f"## Conversation History\n{history_text}\n\n{file_context}"
        )

        try:
            phase1_response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template("agent/dream_phase1.md", strip=True),
                    },
                    {"role": "user", "content": phase1_prompt},
                ],
                tools=None,
                tool_choice=None,
            )
            analysis = phase1_response.content or ""
            logger.debug("Dream Phase 1 analysis ({} chars): {}", len(analysis), analysis[:500])
        except Exception:
            logger.exception("Dream Phase 1 failed")
            return False

        # Phase 2: Delegate to AgentRunner with read_file / edit_file
        existing_skills = self._list_existing_skills()
        skills_section = ""
        if existing_skills:
            skills_section = (
                "\n\n## Existing Skills\n"
                + "\n".join(f"- {s}" for s in existing_skills)
            )
        phase2_prompt = f"## Analysis Result\n{analysis}\n\n{file_context}{skills_section}"

        tools = self._tools
        skill_creator_path = BUILTIN_SKILLS_DIR / "skill-creator" / "SKILL.md"
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": render_template(
                    "agent/dream_phase2.md",
                    strip=True,
                    skill_creator_path=str(skill_creator_path),
                ),
            },
            {"role": "user", "content": phase2_prompt},
        ]

        try:
            result = await self._runner.run(AgentRunSpec(
                initial_messages=messages,
                tools=tools,
                model=self.model,
                max_iterations=self.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                fail_on_tool_error=False,
            ))
            logger.debug(
                "Dream Phase 2 complete: stop_reason={}, tool_events={}",
                result.stop_reason, len(result.tool_events),
            )
            for ev in (result.tool_events or []):
                logger.info("Dream tool_event: name={}, status={}, detail={}", ev.get("name"), ev.get("status"), ev.get("detail", "")[:200])
        except Exception:
            logger.exception("Dream Phase 2 failed")
            result = None

        # Build changelog from tool events
        changelog: list[str] = []
        if result and result.tool_events:
            for event in result.tool_events:
                if event["status"] == "ok":
                    changelog.append(f"{event['name']}: {event['detail']}")

        # Advance cursor — always, to avoid re-processing Phase 1
        new_cursor = batch[-1]["cursor"]
        self.store.set_last_dream_cursor(new_cursor)
        self.store.compact_history()

        if result and result.stop_reason == "completed":
            logger.info(
                "Dream done: {} change(s), cursor advanced to {}",
                len(changelog), new_cursor,
            )
        else:
            reason = result.stop_reason if result else "exception"
            logger.warning(
                "Dream incomplete ({}): cursor advanced to {}",
                reason, new_cursor,
            )

        # Git auto-commit (only when there are actual changes)
        if changelog and self.store.git.is_initialized():
            ts = batch[-1]["timestamp"]
            sha = self.store.git.auto_commit(f"dream: {ts}, {len(changelog)} change(s)")
            if sha:
                logger.info("Dream commit: {}", sha)

        return True
