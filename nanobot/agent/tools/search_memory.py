"""Search memory tool: full-text + semantic search over conversation history."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from nanobot.agent.embedding import EmbeddingService
    from nanobot.agent.memory import MemoryStore

_MAX_CONTENT_DISPLAY = 200
_MAX_RESULTS = 20


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("Search query — keywords or a natural-language description"),
        limit=IntegerSchema(
            description="Maximum number of results to return (default 5, max 20)",
        ),
        required=["query"],
    )
)
class SearchMemoryTool(Tool):
    """Search conversation history via keyword (FTS5) + semantic (embedding) hybrid."""

    def __init__(
        self,
        store: "MemoryStore",
        embedding: "EmbeddingService | None" = None,
    ):
        self._store = store
        self._embedding = embedding

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return (
            "Search past conversation history. Accepts either keywords or a "
            "natural-language description of what to find (semantic search "
            "is used when available). Use when the user references previous "
            "conversations or you need to recall past context."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, query: str, limit: int = 5, **kwargs: Any) -> str:
        limit = min(max(1, limit), _MAX_RESULTS)
        hybrid = self._embedding is not None and self._store.vec_available
        try:
            if hybrid:
                results = await self._store.hybrid_search(
                    query, self._embedding, limit=limit,
                )
            else:
                results = self._store.search_history(query, limit=limit)
        except Exception as e:
            return f"Error searching memory: {e}"

        if not results:
            return f'No results found for "{query}".'

        mode_label = "hybrid" if hybrid else "keyword"
        lines = [f'Found {len(results)} result(s) for "{query}" ({mode_label}):\n']
        for r in results:
            content = r["content"]
            if len(content) > _MAX_CONTENT_DISPLAY:
                content = content[:_MAX_CONTENT_DISPLAY] + "..."
            src = r.get("source", "")
            src_label = f" [{src}]" if src else ""
            if "score" in r:
                meta = f" (score: {r['score']:.3f})"
            elif "rank" in r:
                meta = f" (relevance: {-r['rank']:.2f})"
            else:
                meta = ""
            lines.append(f"[{r['timestamp']}]{src_label}{meta}")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).rstrip()
