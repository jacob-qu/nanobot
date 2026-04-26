"""Tests for ReconcileEngine."""

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.memory_index import MemoryIndex
from nanobot.agent.reconcile import ReconcileEngine


def _pad_vec(vals: list[float], dim: int = 1536) -> bytes:
    padded = list(vals) + [0.0] * (dim - len(vals))
    return struct.pack(f"{dim}f", *padded)


@pytest.fixture
def index(tmp_path: Path) -> MemoryIndex:
    return MemoryIndex(db_path=tmp_path / "index.db", embedding_dim=1536)


class _FakeEmbedding:
    """Deterministic embedding: hash content → vector."""
    dimensions = 1536
    def __init__(self):
        self.calls: list[list[str]] = []
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for t in texts:
            h = abs(hash(t)) % 1000
            out.append([h / 1000.0] + [0.0] * (self.dimensions - 1))
        return out


class _FakeLLM:
    """Returns pre-canned JSON strings."""
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []
    async def complete(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0) if self._responses else "[]"


class TestBootstrap:
    @pytest.mark.asyncio
    async def test_bootstrap_parses_md_and_creates_items(
        self, index: MemoryIndex, tmp_path: Path,
    ):
        md = "## Demo\n- rule 1\n- rule 2\n"
        md_file = tmp_path / "MEMORY.md"
        md_file.write_text(md)

        llm = _FakeLLM(responses=[
            # concept assignment
            json.dumps([
                {"item_index": 0, "concepts": [{"new": {"name": "rules", "description": "d"}}]},
                {"item_index": 1, "concepts": [{"new": {"name": "rules", "description": "d"}}]},
            ]),
            # relation inference
            "[]",
        ])

        engine = ReconcileEngine(
            index=index,
            embedding=_FakeEmbedding(),
            llm=llm,
            memory_file=md_file,
            source_file="memory/MEMORY.md",
            threshold=0.92,
        )

        await engine.bootstrap(current_commit="deadbeef")

        items = index.list_items()
        # expect 3 items (heading + 2 list items)
        assert len(items) == 3
        assert index.get_meta("last_reconciled_commit") == "deadbeef"

    @pytest.mark.asyncio
    async def test_bootstrap_emits_no_impact_issues(
        self, index: MemoryIndex, tmp_path: Path,
    ):
        md_file = tmp_path / "MEMORY.md"
        md_file.write_text("## A\n- item\n")
        llm = _FakeLLM(responses=[
            json.dumps([{"item_index": 0, "concepts": [{"new": {"name": "x", "description": "d"}}]}]),
            "[]",
        ])
        engine = ReconcileEngine(
            index=index, embedding=_FakeEmbedding(), llm=llm,
            memory_file=md_file, source_file="memory/MEMORY.md", threshold=0.92,
        )
        await engine.bootstrap(current_commit="c1")
        assert index.list_open_issues() == []


class TestIncrementalRun:
    @pytest.mark.asyncio
    async def test_run_no_changes_is_noop(
        self, index: MemoryIndex, tmp_path: Path,
    ):
        md = "## A\n- item\n"
        md_file = tmp_path / "MEMORY.md"
        md_file.write_text(md)

        llm = _FakeLLM(responses=[
            json.dumps([{"item_index": 0, "concepts": [{"new": {"name": "x", "description": "d"}}]}]),
            "[]",
        ])
        engine = ReconcileEngine(
            index=index, embedding=_FakeEmbedding(), llm=llm,
            memory_file=md_file, source_file="memory/MEMORY.md", threshold=0.92,
        )
        # bootstrap
        await engine.bootstrap(current_commit="c1")
        # run with no content change
        called_before = len(llm.calls)
        changes = await engine.run(previous_content=md, trigger_ref="c2")
        # no chunks changed → no new LLM calls
        assert len(llm.calls) == called_before
        assert changes.total_changes == 0

    @pytest.mark.asyncio
    async def test_run_modified_item_processes_changes(
        self, index: MemoryIndex, tmp_path: Path,
    ):
        md_v1 = "## A\n- rule: X\n- rule: Y\n"
        md_v2 = "## A\n- rule: X (updated)\n- rule: Y\n"
        md_file = tmp_path / "MEMORY.md"
        md_file.write_text(md_v1)

        llm = _FakeLLM(responses=[
            # bootstrap concept
            json.dumps([
                {"item_index": 0, "concepts": [{"new": {"name": "h", "description": "d"}}]},
                {"item_index": 1, "concepts": []},
                {"item_index": 2, "concepts": []},
            ]),
            "[]",  # bootstrap relations
            # run concept assignment
            json.dumps([{"item_index": 0, "concepts": []}]),
            "[]",  # run relations
            "[]",  # impact review — no affected
        ])

        engine = ReconcileEngine(
            index=index, embedding=_FakeEmbedding(), llm=llm,
            memory_file=md_file, source_file="memory/MEMORY.md", threshold=0.92,
        )
        await engine.bootstrap(current_commit="c1")

        # simulate edit
        md_file.write_text(md_v2)
        changes = await engine.run(previous_content=md_v1, trigger_ref="c2")

        # at least 1 change should be processed (the modified rule item)
        assert changes.total_changes >= 1
