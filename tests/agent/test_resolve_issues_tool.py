"""Tests for ResolveIssuesTool — agent-side batch close for Dream alerts."""

from pathlib import Path

import pytest

from nanobot.agent.memory_index import ConsistencyIssue, MemoryIndex
from nanobot.agent.tools.memory_query import ListOpenIssuesTool, ResolveIssuesTool


@pytest.fixture
def index(tmp_path: Path) -> MemoryIndex:
    return MemoryIndex(db_path=tmp_path / "index.db", embedding_dim=1536)


def _seed(index: MemoryIndex, a: str, b: str, created_at: int = 1000) -> str:
    return index.add_issue(ConsistencyIssue(
        id="", trigger_event="dream_scan", trigger_ref=None,
        issue_type="impact_unreviewed",
        subject_ids=f'[{{"kind":"item","id":"{a}"}},'
                    f'{{"kind":"item","id":"{b}"}}]',
        description="desc", severity="medium", status="open",
        resolution=None, created_at=created_at, resolved_at=None,
    ))


class TestResolveIssuesTool:
    @pytest.mark.asyncio
    async def test_resolves_single(self, index: MemoryIndex):
        issue_id = _seed(index, "a", "b")
        tool = ResolveIssuesTool(index)
        out = await tool.execute(
            issue_ids=[issue_id], resolution="已确认两个条目可合并", status="resolved",
        )
        assert "已关闭 1 条" in out
        assert index.list_open_issues() == []

    @pytest.mark.asyncio
    async def test_resolves_multiple(self, index: MemoryIndex):
        ids = [_seed(index, "a", "b"), _seed(index, "c", "d"), _seed(index, "e", "f")]
        tool = ResolveIssuesTool(index)
        out = await tool.execute(
            issue_ids=ids, resolution="批量处理完毕", status="resolved",
        )
        assert "已关闭 3 条" in out
        assert index.list_open_issues() == []

    @pytest.mark.asyncio
    async def test_wontfix_status(self, index: MemoryIndex):
        issue_id = _seed(index, "a", "b")
        tool = ResolveIssuesTool(index)
        out = await tool.execute(
            issue_ids=[issue_id], resolution="本来就这样，不处理", status="wontfix",
        )
        assert "wontfix" in out
        cur = index._db.execute(
            "SELECT status, resolution FROM consistency_issues WHERE id=?",
            (issue_id,),
        )
        row = cur.fetchone()
        assert row["status"] == "wontfix"
        assert row["resolution"] == "本来就这样，不处理"

    @pytest.mark.asyncio
    async def test_empty_resolution_rejected(self, index: MemoryIndex):
        issue_id = _seed(index, "a", "b")
        tool = ResolveIssuesTool(index)
        out = await tool.execute(
            issue_ids=[issue_id], resolution="   ", status="resolved",
        )
        assert "resolution" in out.lower()
        assert len(index.list_open_issues()) == 1

    @pytest.mark.asyncio
    async def test_invalid_status_rejected(self, index: MemoryIndex):
        issue_id = _seed(index, "a", "b")
        tool = ResolveIssuesTool(index)
        out = await tool.execute(
            issue_ids=[issue_id], resolution="ok", status="bogus",
        )
        assert "status" in out.lower()
        assert len(index.list_open_issues()) == 1

    @pytest.mark.asyncio
    async def test_mixed_valid_invalid(self, index: MemoryIndex):
        ok_id = _seed(index, "a", "b")
        closed_id = _seed(index, "c", "d")
        index.resolve_issue(closed_id, "resolved", "earlier")
        tool = ResolveIssuesTool(index)
        out = await tool.execute(
            issue_ids=[ok_id, closed_id, "nonexistent"],
            resolution="部分处理",
            status="resolved",
        )
        assert "已关闭 1 条" in out
        assert "跳过" in out
        assert "nonexistent" in out
        assert closed_id in out
        assert index.list_open_issues() == []

    @pytest.mark.asyncio
    async def test_empty_ids_rejected(self, index: MemoryIndex):
        tool = ResolveIssuesTool(index)
        out = await tool.execute(
            issue_ids=[], resolution="foo", status="resolved",
        )
        assert "issue_ids" in out.lower() or "空" in out


class TestListOpenIssuesOutput:
    @pytest.mark.asyncio
    async def test_seen_count_shown_when_greater_than_one(
        self, index: MemoryIndex,
    ):
        issue_id = _seed(index, "a", "b", created_at=1000)
        # simulate dedup hit
        index.bump_issue_seen(issue_id, now=2000)
        index.bump_issue_seen(issue_id, now=3000)
        tool = ListOpenIssuesTool(index)
        out = await tool.execute(severity="low")
        assert "连续 3 轮" in out

    @pytest.mark.asyncio
    async def test_first_seen_omits_seen_count_line(self, index: MemoryIndex):
        _seed(index, "a", "b")
        tool = ListOpenIssuesTool(index)
        out = await tool.execute(severity="low")
        assert "连续" not in out

    @pytest.mark.asyncio
    async def test_hints_resolve_issues(self, index: MemoryIndex):
        _seed(index, "a", "b")
        tool = ListOpenIssuesTool(index)
        out = await tool.execute(severity="low")
        assert "resolve_issues" in out
