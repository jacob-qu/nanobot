"""Tests for memory query tools."""

import time
from pathlib import Path

import pytest

from nanobot.agent.memory_index import (
    Concept, ItemRecord, MemoryIndex, Relation,
)
from nanobot.agent.tools.memory_query import (
    GetMemoryConceptTool, ListOpenIssuesTool, QueryMemoryImpactTool,
)


@pytest.fixture
def index(tmp_path: Path) -> MemoryIndex:
    return MemoryIndex(db_path=tmp_path / "index.db", embedding_dim=1536)


class TestGetMemoryConceptTool:
    @pytest.mark.asyncio
    async def test_found(self, index: MemoryIndex):
        now = int(time.time())
        c_id = index.upsert_concept(Concept(
            id="", name="三档待办", description="d",
            centroid_embed=None, member_count=0,
            created_at=now, updated_at=now, merged_into=None,
        ))
        item_id = index.upsert_item(ItemRecord(
            id="", source_file="memory/MEMORY.md",
            section_path="A", item_type="list_item",
            content="foo", content_hash="h",
            embedding=None, created_at=now, updated_at=now, removed_at=None,
        ))
        index.link_item_concept(item_id, c_id, 0.9, "llm")

        tool = GetMemoryConceptTool(index)
        result = await tool.execute(query="三档")
        assert "三档待办" in result and "foo" in result

    @pytest.mark.asyncio
    async def test_not_found(self, index: MemoryIndex):
        tool = GetMemoryConceptTool(index)
        result = await tool.execute(query="不存在")
        assert "no concept" in result.lower() or "未找到" in result


class TestQueryMemoryImpactTool:
    @pytest.mark.asyncio
    async def test_reports_incoming_edges(self, index: MemoryIndex):
        now = int(time.time())
        # build X depends_on Y
        x = index.upsert_item(ItemRecord(
            id="", source_file="memory/MEMORY.md", section_path="s",
            item_type="list_item", content="X content", content_hash="hx",
            embedding=None, created_at=now, updated_at=now, removed_at=None,
        ))
        y = index.upsert_item(ItemRecord(
            id="", source_file="memory/MEMORY.md", section_path="s",
            item_type="list_item", content="Y target", content_hash="hy",
            embedding=None, created_at=now, updated_at=now, removed_at=None,
        ))
        index.add_relation(Relation(
            id="", from_kind="item", from_id=x, to_kind="item", to_id=y,
            relation_type="depends_on", confidence=0.9, source="llm",
            rationale=None, created_at=now, invalidated_at=None,
        ))
        tool = QueryMemoryImpactTool(index)
        result = await tool.execute(target="Y target")
        assert "X content" in result


class TestListOpenIssuesTool:
    @pytest.mark.asyncio
    async def test_filters_by_severity(self, index: MemoryIndex):
        from nanobot.agent.memory_index import ConsistencyIssue
        now = int(time.time())
        index.add_issue(ConsistencyIssue(
            id="", trigger_event="edit", trigger_ref=None,
            issue_type="impact_unreviewed", subject_ids="[]",
            description="low one", severity="low", status="open",
            resolution=None, created_at=now, resolved_at=None,
        ))
        index.add_issue(ConsistencyIssue(
            id="", trigger_event="edit", trigger_ref=None,
            issue_type="impact_unreviewed", subject_ids="[]",
            description="high one", severity="high", status="open",
            resolution=None, created_at=now, resolved_at=None,
        ))
        tool = ListOpenIssuesTool(index)
        r_all = await tool.execute(severity="low")
        assert "low one" in r_all and "high one" in r_all
        r_high = await tool.execute(severity="high")
        assert "high one" in r_high and "low one" not in r_high


class TestTriggerDreamTool:
    @pytest.mark.asyncio
    async def test_execute_awaits_dream_and_reports_result(self):
        from nanobot.agent.tools.memory_query import TriggerDreamTool

        calls: list[int] = []

        class _FakeDream:
            reconcile_engine = None  # no engine → tool reports minimal

            async def run(self):
                calls.append(1)
                return True

        tool = TriggerDreamTool(_FakeDream())
        result = await tool.execute(reason="test reason")
        # Tool synchronously awaits Dream, then reports
        assert calls == [1]
        assert "Dream run 完成" in result
        assert "did_work=True" in result
        assert "test reason" in result

    @pytest.mark.asyncio
    async def test_execute_reports_reconcile_outcome(self, tmp_path: Path):
        from nanobot.agent.memory_index import ConsistencyIssue, MemoryIndex
        from nanobot.agent.tools.memory_query import TriggerDreamTool

        idx = MemoryIndex(db_path=tmp_path / "idx.db", embedding_dim=1536)
        # Seed 2 open issues
        now = int(time.time())
        idx.add_issue(ConsistencyIssue(
            id="", trigger_event="edit", trigger_ref=None,
            issue_type="impact_unreviewed", subject_ids="[]",
            description="one", severity="medium", status="open",
            resolution=None, created_at=now, resolved_at=None,
        ))
        idx.add_issue(ConsistencyIssue(
            id="", trigger_event="edit", trigger_ref=None,
            issue_type="impact_unreviewed", subject_ids="[]",
            description="two", severity="high", status="open",
            resolution=None, created_at=now, resolved_at=None,
        ))
        idx.set_meta("last_reconciled_commit", "abcdef1234567890")

        class _FakeEngine:
            def __init__(self, index):
                self._index = index

        class _FakeDream:
            def __init__(self, engine):
                self.reconcile_engine = engine

            async def run(self):
                return False  # no-op, issues were pre-seeded

        tool = TriggerDreamTool(_FakeDream(_FakeEngine(idx)))
        result = await tool.execute()
        assert "Dream 完成" in result
        assert "告警 2 条" in result
        assert "abcdef12" in result  # watermark short sha

    @pytest.mark.asyncio
    async def test_execute_swallows_dream_exceptions(self):
        from nanobot.agent.tools.memory_query import TriggerDreamTool

        class _BadDream:
            reconcile_engine = None

            async def run(self):
                raise RuntimeError("boom")

        tool = TriggerDreamTool(_BadDream())
        result = await tool.execute()
        # Error is reported, not raised
        assert "Dream 执行失败" in result and "boom" in result
