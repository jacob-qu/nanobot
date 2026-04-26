"""ReconcileEngine — diff-driven pipeline from MEMORY.md to MemoryIndex."""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from nanobot.agent.md_chunker import Chunk, chunk_markdown
from nanobot.agent.memory_index import (
    Concept,
    ConsistencyIssue,
    ItemRecord,
    MemoryIndex,
    Relation,
)
from nanobot.agent.reconcile_diff import DiffResult, align_items
from nanobot.utils.prompt_templates import render_template


class _LLMLike(Protocol):
    async def complete(self, prompt: str, **kwargs: Any) -> str: ...


class _EmbeddingLike(Protocol):
    dimensions: int
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _parse_json_array(text: str) -> list[dict]:
    """Tolerant JSON parsing: strip markdown fences, return [] on failure."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError as e:
        logger.warning(f"reconcile: JSON parse failed: {e}; text head: {text[:200]}")
        return []


class ReconcileEngine:
    """Drive the index update pipeline from MEMORY.md diffs."""

    def __init__(
        self,
        index: MemoryIndex,
        embedding: _EmbeddingLike,
        llm: _LLMLike,
        memory_file: Path,
        source_file: str,
        threshold: float = 0.92,
        concept_top_k: int = 5,
        relation_top_k: int = 5,
    ):
        self._index = index
        self._embedding = embedding
        self._llm = llm
        self._memory_file = memory_file
        self._source_file = source_file
        self._threshold = threshold
        self._concept_top_k = concept_top_k
        self._relation_top_k = relation_top_k

    # -- bootstrap --------------------------------------------------------

    async def bootstrap(self, current_commit: str) -> None:
        """Initial build from current MEMORY.md. No impact issues emitted."""
        md = self._memory_file.read_text(encoding="utf-8") if self._memory_file.exists() else ""
        chunks = chunk_markdown(md)
        if not chunks:
            self._index.set_meta("last_reconciled_commit", current_commit)
            return

        # 1. Embed all
        vecs = await self._embedding.embed_batch([c.content for c in chunks])

        # 2. Insert items
        now = int(time.time())
        item_ids: list[str] = []
        for c, vec in zip(chunks, vecs):
            item = ItemRecord(
                id="",
                source_file=self._source_file,
                section_path=c.section_path,
                item_type=c.item_type,
                content=c.content,
                content_hash=c.content_hash,
                embedding=_pack(vec),
                created_at=now,
                updated_at=now,
                removed_at=None,
            )
            item_ids.append(self._index.upsert_item(item))

        # 3. Concept assignment (one batched LLM call)
        await self._assign_concepts(chunks, item_ids)

        # 4. Relation inference (one batched LLM call)
        await self._infer_relations(chunks, item_ids)

        # 5. Bootstrap mode: no impact issues
        self._index.set_meta("last_reconciled_commit", current_commit)

    # -- concept + relation helpers (shared with run()) ------------------

    async def _assign_concepts(
        self, chunks: list[Chunk], item_ids: list[str],
    ) -> None:
        if not chunks:
            return
        # existing concepts
        existing = [
            {"id": c.id, "name": c.name, "description": c.description or ""}
            for c in self._all_live_concepts()
        ]
        items_json = json.dumps(
            [{"index": i, "section": c.section_path, "content": c.content}
             for i, c in enumerate(chunks)],
            ensure_ascii=False,
        )
        existing_json = json.dumps(existing, ensure_ascii=False)
        prompt = render_template(
            "agent/reconcile_concepts.md",
            items=items_json,
            existing_concepts=existing_json,
        )
        raw = await self._llm.complete(prompt)
        assignments = _parse_json_array(raw)

        for a in assignments:
            idx = a.get("item_index")
            if not isinstance(idx, int) or idx >= len(item_ids):
                continue
            item_id = item_ids[idx]
            item_content = chunks[idx].content
            for c_spec in a.get("concepts", []):
                concept_id = self._resolve_concept_spec(c_spec, item_content)
                if concept_id:
                    self._index.link_item_concept(
                        item_id, concept_id, confidence=0.8, source="llm",
                    )

    def _resolve_concept_spec(
        self, spec: dict[str, Any], item_content: str,
    ) -> str | None:
        if "existing_id" in spec:
            return spec["existing_id"]
        new = spec.get("new")
        if not new:
            return None
        now = int(time.time())
        concept = Concept(
            id="", name=new.get("name", "unnamed"),
            description=new.get("description"),
            centroid_embed=None, member_count=0,
            created_at=now, updated_at=now, merged_into=None,
        )
        return self._index.upsert_concept(concept)

    def _all_live_concepts(self) -> list[Concept]:
        cur = self._index._db.execute(
            "SELECT * FROM concepts WHERE merged_into IS NULL"
        )
        from nanobot.agent.memory_index import _row_to_concept
        return [_row_to_concept(r) for r in cur.fetchall()]

    async def _infer_relations(
        self, chunks: list[Chunk], item_ids: list[str],
    ) -> None:
        if not chunks:
            return
        candidates: list[dict[str, Any]] = []
        for i, c in enumerate(chunks):
            candidates.append({
                "index": i,
                "section": c.section_path,
                "content": c.content[:500],
            })

        # Gather siblings: alive items from index minus these newly-added ones
        sibling_set: list[dict[str, Any]] = []
        mine = set(item_ids)
        for it in self._index.list_items(alive=True):
            if it.id in mine:
                continue
            sibling_set.append({"id": it.id, "section": it.section_path,
                                 "content": it.content[:500]})
            if len(sibling_set) >= 50:  # cap to avoid huge prompt
                break

        prompt = render_template(
            "agent/reconcile_relations.md",
            candidates=json.dumps(candidates, ensure_ascii=False),
            siblings=json.dumps(sibling_set, ensure_ascii=False),
        )
        raw = await self._llm.complete(prompt)
        edges = _parse_json_array(raw)

        now = int(time.time())
        for e in edges:
            fi = e.get("from_index")
            to_id = e.get("to_id")
            if not isinstance(fi, int) or not to_id or fi >= len(item_ids):
                continue
            rel = Relation(
                id="",
                from_kind="item", from_id=item_ids[fi],
                to_kind="item", to_id=to_id,
                relation_type=e.get("relation_type", "references"),
                confidence=float(e.get("confidence", 0.5)),
                source="llm", rationale=e.get("rationale"),
                created_at=now, invalidated_at=None,
            )
            self._index.add_relation(rel)
