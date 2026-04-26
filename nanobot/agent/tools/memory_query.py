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
            return f'未找到匹配 "{query}" 的概念。'
        lines: list[str] = []
        for c in matches[:3]:
            lines.append(f"# 概念：{c.name}")
            if c.description:
                lines.append(f"> {c.description}")
            lines.append("")
            item_ids = self._index.list_items_for_concept(c.id)
            if not item_ids:
                lines.append("（该概念下暂无条目）")
            else:
                lines.append(f"## 覆盖的记忆条目（{len(item_ids)}）")
                for iid in item_ids[:15]:
                    item = self._index.get_item(iid)
                    if not item:
                        continue
                    snippet = item.content.replace("\n", " ⏎ ")
                    if len(snippet) > 150:
                        snippet = snippet[:150] + "…"
                    lines.append(f"- [{item.section_path}]")
                    lines.append(f"  {snippet}")
                if len(item_ids) > 15:
                    lines.append(f"...（另有 {len(item_ids) - 15} 条未显示）")
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
        """Aggregate impact across all items carrying this concept.

        Output groups dependents by relation_type and deduplicates by from_id
        so the same source only shows once even if it touches multiple members.
        """
        lines = [
            f"# 概念「{concept.name}」的影响面",
            f"（覆盖 {len(member_item_ids)} 个条目）",
            "",
        ]

        # Collect incoming edges across all members, dedup by from_id
        by_type: dict[str, dict[str, tuple[float, str | None]]] = {}
        for item_id in member_item_ids:
            for e in self._index.relations_to("item", item_id):
                bucket = by_type.setdefault(e.relation_type, {})
                prior = bucket.get(e.from_id)
                # Keep the highest-confidence edge per source
                if prior is None or e.confidence > prior[0]:
                    bucket[e.from_id] = (e.confidence, e.rationale)

        for e in concept_impact.incoming_edges:
            bucket = by_type.setdefault(e.relation_type, {})
            prior = bucket.get(e.from_id)
            if prior is None or e.confidence > prior[0]:
                bucket[e.from_id] = (e.confidence, e.rationale)

        if not by_type:
            lines.append("（未发现引用或依赖）")
            return "\n".join(lines)

        # Human-friendly type labels
        type_labels = {
            "references": "引用",
            "depends_on": "依赖",
            "supersedes": "取代",
            "conflicts_with": "冲突",
            "implements": "实现",
            "documents": "描述",
        }
        # Render, relation types ordered by importance
        order = ["depends_on", "conflicts_with", "references", "implements",
                 "documents", "supersedes"]
        for rtype in order:
            if rtype not in by_type:
                continue
            bucket = by_type[rtype]
            label = type_labels.get(rtype, rtype)
            lines.append(f"## {label}（{rtype}）")
            # Sort by confidence desc, cap at 8 entries
            entries = sorted(bucket.items(), key=lambda kv: -kv[1][0])[:8]
            for from_id, (conf, rationale) in entries:
                subj = self._resolve_label("item", from_id)
                lines.append(f"- {subj}  (conf {conf:.2f})")
                if rationale:
                    lines.append(f"  └ {rationale}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _format_impact(self, kind: str, label: str, impact: "ImpactResult") -> str:
        lines = [f"# 条目「{label}」的影响面"]
        if not impact.incoming_edges and not impact.transitive_nodes:
            lines.append("")
            lines.append("（未发现引用或依赖）")
            return "\n".join(lines)
        type_labels = {
            "references": "引用",
            "depends_on": "依赖",
            "supersedes": "取代",
            "conflicts_with": "冲突",
            "implements": "实现",
            "documents": "描述",
        }
        by_type: dict[str, list] = {}
        for e in impact.incoming_edges:
            by_type.setdefault(e.relation_type, []).append(e)
        lines.append("")
        for rtype, edges in by_type.items():
            human = type_labels.get(rtype, rtype)
            lines.append(f"## {human}（{rtype}）")
            edges.sort(key=lambda e: -e.confidence)
            for e in edges[:8]:
                subj = self._resolve_label(e.from_kind, e.from_id)
                lines.append(f"- {subj}  (conf {e.confidence:.2f})")
                if e.rationale:
                    lines.append(f"  └ {e.rationale}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _resolve_label(self, kind: str, id: str) -> str:
        if kind == "item":
            it = self._index.get_item(id)
            if it:
                # Collapse multiline content to single-line preview
                preview = it.content.replace("\n", " ⏎ ")
                if len(preview) > 70:
                    preview = preview[:70] + "…"
                return f"[{it.section_path}] {preview}"
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
            return f"没有待处理的一致性告警（severity ≥ {severity}）。"
        lines = [f"# 未处理的记忆一致性告警（{len(issues)} 条，severity ≥ {severity}）", ""]
        for i in issues:
            lines.append(f"## [{i.severity}] {i.issue_type}")
            lines.append(i.description)
            lines.append("")
        return "\n".join(lines).rstrip()
