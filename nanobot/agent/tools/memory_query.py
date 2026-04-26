"""Agent tools for querying MemoryIndex."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from nanobot.agent.memory_index import ImpactResult, MemoryIndex


@tool_parameters(tool_parameters_schema(
    query=StringSchema("Concept name (fuzzy match) or semantic phrase"),
    required=["query"],
))
class GetMemoryConceptTool(Tool):
    """Fetch a concept's definition + member items by name."""

    def __init__(self, index: "MemoryIndex"):
        self._index = index

    @property
    def name(self) -> str:
        return "get_memory_concept"

    @property
    def description(self) -> str:
        return (
            "Look up a memory concept by name (fuzzy). Returns the concept's "
            "description and all member memory items with their section path. "
            "Use when the user asks about a named concept or policy in memory."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, query: str, **kwargs: Any) -> str:
        matches = self._index.find_concept_by_name(query, fuzzy=True)
        if not matches:
            return f'No concept found matching "{query}".'
        lines: list[str] = []
        for c in matches[:3]:
            lines.append(f"## {c.name}")
            if c.description:
                lines.append(c.description)
            item_ids = self._index.list_items_for_concept(c.id)
            for iid in item_ids[:20]:
                item = self._index.get_item(iid)
                if not item:
                    continue
                snippet = item.content if len(item.content) <= 200 else item.content[:200] + "..."
                lines.append(f"- [{item.section_path}] {snippet}")
            lines.append("")
        return "\n".join(lines).rstrip()


@tool_parameters(tool_parameters_schema(
    target=StringSchema(
        "Item content fragment or concept name. The tool will locate the "
        "matching item/concept and return everything that depends on or references it."
    ),
    required=["target"],
))
class QueryMemoryImpactTool(Tool):
    """Reverse-trace what other memory items depend on or reference a given target."""

    def __init__(self, index: "MemoryIndex"):
        self._index = index

    @property
    def name(self) -> str:
        return "query_memory_impact"

    @property
    def description(self) -> str:
        return (
            "Given a memory item content fragment or concept name, list all "
            "other items/concepts that reference or depend on it. Use before "
            "modifying a memory to understand what downstream items may be affected."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, target: str, **kwargs: Any) -> str:
        # Try concept match first; if found, union impact across all member items
        concepts = self._index.find_concept_by_name(target, fuzzy=True)
        if concepts:
            concept = concepts[0]
            member_item_ids = self._index.list_items_for_concept(concept.id)
            # Also union in direct concept-level relations (rare, but support them)
            concept_impact = self._index.query_impact("concept", concept.id, depth=2)
            return self._format_concept_impact(concept, member_item_ids, concept_impact)

        # Fall back to item content fuzzy search (simple LIKE)
        cur = self._index._db.execute(
            "SELECT id FROM memory_items WHERE content LIKE ? AND removed_at IS NULL LIMIT 1",
            (f"%{target}%",),
        )
        row = cur.fetchone()
        if not row:
            return f'No memory item or concept matching "{target}".'
        impact = self._index.query_impact("item", row["id"], depth=2)
        item = self._index.get_item(row["id"])
        return self._format_impact("item", item.content[:60] if item else target, impact)

    def _format_concept_impact(
        self, concept, member_item_ids: list[str], concept_impact,
    ) -> str:
        """Aggregate impact across all items carrying this concept."""
        lines = [f"Impact of concept `{concept.name}` (covers {len(member_item_ids)} item(s)):"]

        # Gather all incoming edges across member items
        seen_pairs: set[tuple[str, str]] = set()  # dedup (from_id, to_id)
        rows: list[str] = []
        for item_id in member_item_ids:
            item = self._index.get_item(item_id)
            if not item:
                continue
            item_label = f"[{item.section_path}] {item.content[:50]}"
            for e in self._index.relations_to("item", item_id):
                key = (e.from_id, e.to_id)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                subj = self._resolve_label(e.from_kind, e.from_id)
                rows.append(
                    f"- {subj} --[{e.relation_type}, {e.confidence:.2f}]--> {item_label}"
                )
                if e.rationale:
                    rows.append(f"    rationale: {e.rationale}")

        # Also include any direct concept-level relations (if the graph ever grows them)
        for e in concept_impact.incoming_edges:
            subj = self._resolve_label(e.from_kind, e.from_id)
            rows.append(
                f"- {subj} --[{e.relation_type}, {e.confidence:.2f}]--> concept:{concept.name}"
            )
            if e.rationale:
                rows.append(f"    rationale: {e.rationale}")

        if not rows:
            lines.append("  (no dependents found)")
        else:
            lines.extend(rows)
        return "\n".join(lines)

    def _format_impact(self, kind: str, label: str, impact: "ImpactResult") -> str:
        lines = [f"Impact of {kind} `{label}`:"]
        if not impact.incoming_edges and not impact.transitive_nodes:
            lines.append("  (no dependents found)")
            return "\n".join(lines)
        for e in impact.incoming_edges:
            subj = self._resolve_label(e.from_kind, e.from_id)
            lines.append(
                f"- {subj} --[{e.relation_type}, {e.confidence:.2f}]--> {label}"
            )
            if e.rationale:
                lines.append(f"    rationale: {e.rationale}")
        for (fk, fid, depth) in impact.transitive_nodes:
            subj = self._resolve_label(fk, fid)
            lines.append(f"  (depth {depth}) {subj}")
        return "\n".join(lines)

    def _resolve_label(self, kind: str, id: str) -> str:
        if kind == "item":
            it = self._index.get_item(id)
            if it:
                return f"[{it.section_path}] {it.content[:60]}"
        elif kind == "concept":
            c = self._index.get_concept(id)
            if c:
                return f"concept:{c.name}"
        return f"{kind}:{id}"


@tool_parameters(tool_parameters_schema(
    severity=StringSchema(
        'Minimum severity filter: "low" | "medium" | "high" (default "medium")'
    ),
))
class ListOpenIssuesTool(Tool):
    """List currently open memory consistency issues."""

    def __init__(self, index: "MemoryIndex"):
        self._index = index

    @property
    def name(self) -> str:
        return "list_open_issues"

    @property
    def description(self) -> str:
        return (
            "List open memory-consistency issues (awaiting user confirmation / "
            "auto-fix). Use when the user asks about pending memory issues."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, severity: str = "medium", **kwargs: Any) -> str:
        issues = self._index.list_open_issues(severity_min=severity)
        if not issues:
            return f"No open issues at severity>={severity}."
        lines = [f"{len(issues)} open issue(s) at severity>={severity}:"]
        for i in issues:
            lines.append(f"- [{i.severity}] {i.issue_type}: {i.description}")
        return "\n".join(lines)
