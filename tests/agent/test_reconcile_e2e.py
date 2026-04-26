"""End-to-end smoke test: bootstrap a tiny MEMORY.md then modify one item."""

import json
from pathlib import Path

import pytest

from nanobot.agent.memory_index import MemoryIndex
from nanobot.agent.reconcile import ReconcileEngine


class _FakeEmbed:
    dimensions = 1536
    async def embed_batch(self, texts):
        return [[float(hash(t) % 1000) / 1000.0] + [0.0] * 1535 for t in texts]


class _FakeLLM:
    def __init__(self, responses):
        self._r = list(responses)
    async def complete(self, prompt, **k):
        return self._r.pop(0) if self._r else "[]"


@pytest.mark.asyncio
async def test_bootstrap_then_modify_runs_full_pipeline(tmp_path: Path):
    md_v1 = """## 飞书日报
- 三档判断：有回复→已处理；表情→待确认；无回复→待办
- 定时：工作日 22:00
"""
    md_v2 = md_v1.replace(
        "三档判断：有回复→已处理；表情→待确认；无回复→待办",
        "三档判断：有回复→已处理；表情→待确认；无回复→待办；仅点赞→已处理",
    )
    md_file = tmp_path / "MEMORY.md"
    md_file.write_text(md_v1)

    idx = MemoryIndex(db_path=tmp_path / "index.db", embedding_dim=1536)
    llm = _FakeLLM([
        # bootstrap: concept assignment (3 items: 1 heading + 2 list_items)
        json.dumps([
            {"item_index": 0, "concepts": [{"new": {"name": "section", "description": "d"}}]},
            {"item_index": 1, "concepts": [{"new": {"name": "三档判断", "description": "d"}}]},
            {"item_index": 2, "concepts": [{"new": {"name": "定时", "description": "d"}}]},
        ]),
        # bootstrap: relations
        "[]",
        # run: concept on modified
        json.dumps([{"item_index": 0, "concepts": []}]),
        # run: relations
        "[]",
        # run: impact review
        "[]",
    ])
    engine = ReconcileEngine(
        index=idx, embedding=_FakeEmbed(), llm=llm,
        memory_file=md_file, source_file="memory/MEMORY.md",
    )
    await engine.bootstrap(current_commit="c1")
    assert len(idx.list_items()) == 3
    assert idx.get_meta("last_reconciled_commit") == "c1"

    md_file.write_text(md_v2)
    r = await engine.run(previous_content=md_v1, trigger_ref="c2")
    assert r.total_changes >= 1  # at least 1 modified


@pytest.mark.asyncio
async def test_context_builder_includes_memory_tools_hint_when_enabled(tmp_path: Path):
    from nanobot.agent.context import ContextBuilder

    cb = ContextBuilder(
        workspace=tmp_path,
        timezone="UTC",
        memory_index_enabled=True,
    )
    prompt = cb.build_system_prompt()
    assert "Memory Tools" in prompt
    assert "query_memory_impact" in prompt


@pytest.mark.asyncio
async def test_context_builder_omits_memory_tools_hint_when_disabled(tmp_path: Path):
    from nanobot.agent.context import ContextBuilder

    cb = ContextBuilder(
        workspace=tmp_path,
        timezone="UTC",
        memory_index_enabled=False,
    )
    prompt = cb.build_system_prompt()
    assert "Memory Tools" not in prompt
