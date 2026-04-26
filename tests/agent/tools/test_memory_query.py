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
