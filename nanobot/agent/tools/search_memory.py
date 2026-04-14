"""Search memory tool: full-text search over conversation history."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from nanobot.agent.memory import MemoryStore

_MAX_CONTENT_DISPLAY = 200
_MAX_RESULTS = 20


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("Search keywords or phrase"),
        limit=IntegerSchema(
            description="Maximum number of results to return (default 5, max 20)",
        ),
        required=["query"],
    )
)
class SearchMemoryTool(Tool):
    """Tool to search conversation history via full-text search."""

    def __init__(self, store: MemoryStore):
        self._store = store

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return (
            "Search past conversation history. Use when the user references "
            "previous conversations, past requests, or when you need to recall "
            "what was discussed before."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, query: str, limit: int = 5, **kwargs: Any) -> str:
        limit = min(max(1, limit), _MAX_RESULTS)
        try:
            results = self._store.search_history(query, limit=limit)
        except Exception as e:
            return f"Error searching memory: {e}"

        if not results:
            return f'No results found for "{query}".'

        lines = [f'Found {len(results)} result(s) for "{query}":\n']
        for r in results:
            content = r["content"]
            if len(content) > _MAX_CONTENT_DISPLAY:
                content = content[:_MAX_CONTENT_DISPLAY] + "..."
            rank = r.get("rank", 0)
            lines.append(f"[{r['timestamp']}] (relevance: {-rank:.2f})")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).rstrip()
