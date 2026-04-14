"""Tests for the search_memory tool."""

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.search_memory import SearchMemoryTool


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


@pytest.fixture
def tool(store):
    return SearchMemoryTool(store=store)


class TestSearchMemoryTool:
    def test_name(self, tool):
        assert tool.name == "search_memory"

    def test_read_only(self, tool):
        assert tool.read_only is True

    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self, tool, store):
        store.append_history("用户请求编写自动部署脚本")
        store.append_history("修复登录页面样式")
        result = await tool.execute(query="部署")
        assert "部署" in result
        assert "Found" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self, tool, store):
        store.append_history("some content")
        result = await tool.execute(query="不存在xyz")
        assert "No results" in result

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, tool, store):
        for i in range(10):
            store.append_history(f"测试条目 {i}")
        result = await tool.execute(query="测试", limit=2)
        lines = [l for l in result.split("\n") if l.startswith("[")]
        assert len(lines) <= 2

    def test_schema_has_required_query(self, tool):
        schema = tool.parameters
        assert "query" in schema.get("required", [])
