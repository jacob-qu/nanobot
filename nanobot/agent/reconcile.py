"""ReconcileEngine — diff-driven pipeline from MEMORY.md to MemoryIndex."""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
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


@dataclass
class RunResult:
    total_changes: int
    added_count: int
    removed_count: int
    modified_count: int
    issues_created: int


def _chunks_equal(a: list[Chunk], b: list[Chunk]) -> bool:
    if len(a) != len(b):
        return False
    return all(x.content_hash == y.content_hash for x, y in zip(a, b))


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
        valid_sibling_ids = {s["id"] for s in sibling_set}
        for e in edges:
            fi = e.get("from_index")
            if not isinstance(fi, int) or fi >= len(item_ids):
                continue

            # to target: either to_index (within batch) or to_id (sibling)
            to_id: str | None = None
            ti = e.get("to_index")
            if isinstance(ti, int) and 0 <= ti < len(item_ids) and ti != fi:
                to_id = item_ids[ti]
            else:
                raw_to = e.get("to_id")
                if isinstance(raw_to, str) and raw_to in valid_sibling_ids:
                    to_id = raw_to

            if not to_id:
                logger.debug(
                    f"reconcile: dropping relation with invalid target "
                    f"(to_index={ti!r} to_id={e.get('to_id')!r})"
                )
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

    # -- incremental run --------------------------------------------------

    async def run(
        self, previous_content: str, trigger_ref: str | None = None,
    ) -> RunResult:
        """Diff previous_content vs current MEMORY.md, update index, emit issues."""
        current_content = (
            self._memory_file.read_text(encoding="utf-8")
            if self._memory_file.exists() else ""
        )

        old_chunks = chunk_markdown(previous_content)
        new_chunks = chunk_markdown(current_content)

        if _chunks_equal(old_chunks, new_chunks):
            return RunResult(0, 0, 0, 0, 0)

        # Embed only new chunks (old ones have embeddings in the index already)
        new_vecs = await self._embedding.embed_batch([c.content for c in new_chunks])
        new_embeddings_by_hash = {
            c.content_hash: _pack(v) for c, v in zip(new_chunks, new_vecs)
        }

        # Retrieve old-item embeddings from index by content_hash lookup
        old_embeddings_by_hash: dict[str, bytes] = {}
        for oc in old_chunks:
            existing = self._find_item_by_hash(oc.content_hash)
            if existing and existing.embedding:
                old_embeddings_by_hash[oc.content_hash] = existing.embedding

        diff = align_items(
            old=old_chunks, new=new_chunks,
            old_embeddings=old_embeddings_by_hash,
            new_embeddings=new_embeddings_by_hash,
            threshold=self._threshold,
        )

        now = int(time.time())
        new_item_ids: list[str] = []
        changed_chunks: list[Chunk] = []
        changed_ids: list[str] = []

        # Handle added
        for ch in diff.added:
            vec = new_embeddings_by_hash.get(ch.content_hash)
            item = ItemRecord(
                id="", source_file=self._source_file, section_path=ch.section_path,
                item_type=ch.item_type, content=ch.content,
                content_hash=ch.content_hash, embedding=vec,
                created_at=now, updated_at=now, removed_at=None,
            )
            nid = self._index.upsert_item(item)
            new_item_ids.append(nid)
            changed_chunks.append(ch)
            changed_ids.append(nid)

        # Handle removed (tombstone)
        for ch in diff.removed:
            existing = self._find_item_by_hash(ch.content_hash)
            if existing:
                self._index.tombstone_item(existing.id)

        # Handle modified (update existing item in place)
        for pair in diff.modified:
            existing = self._find_item_by_hash(pair.old.content_hash)
            if not existing:
                continue
            vec = new_embeddings_by_hash.get(pair.new.content_hash)
            existing.content = pair.new.content
            existing.content_hash = pair.new.content_hash
            existing.item_type = pair.new.item_type
            existing.section_path = pair.new.section_path
            existing.embedding = vec
            existing.updated_at = now
            self._index.upsert_item(existing)
            changed_chunks.append(pair.new)
            changed_ids.append(existing.id)

        # Ambiguous → write id_remap_ambiguous issue, tombstone olds, insert news
        for grp in diff.ambiguous:
            for ch in grp.olds:
                ex = self._find_item_by_hash(ch.content_hash)
                if ex:
                    self._index.tombstone_item(ex.id)
            for ch in grp.news:
                vec = new_embeddings_by_hash.get(ch.content_hash)
                item = ItemRecord(
                    id="", source_file=self._source_file, section_path=ch.section_path,
                    item_type=ch.item_type, content=ch.content,
                    content_hash=ch.content_hash, embedding=vec,
                    created_at=now, updated_at=now, removed_at=None,
                )
                nid = self._index.upsert_item(item)
                new_item_ids.append(nid)
                changed_chunks.append(ch)
                changed_ids.append(nid)
            self._index.add_issue(ConsistencyIssue(
                id="", trigger_event="dream_scan", trigger_ref=trigger_ref,
                issue_type="id_remap_ambiguous",
                subject_ids=json.dumps(
                    [{"kind": "item", "hash": c.content_hash} for c in grp.olds + grp.news],
                    ensure_ascii=False),
                description=f"ID 对齐歧义 ({grp.kind}): {len(grp.olds)} 老条目 vs {len(grp.news)} 新条目",
                severity="medium", status="open", resolution=None,
                created_at=now, resolved_at=None,
            ))

        # LLM concept + relation on changed set
        if changed_chunks:
            await self._assign_concepts(changed_chunks, changed_ids)
            await self._infer_relations(changed_chunks, changed_ids)

        # Impact review — only for non-new items (existing items that got modified)
        issues_created = 0
        new_item_id_set = set(new_item_ids)
        for item_id in changed_ids:
            if item_id in new_item_id_set:
                continue  # new items have no prior impact graph
            issues_created += await self._review_impact(item_id, trigger_ref, now)

        return RunResult(
            total_changes=len(diff.added) + len(diff.removed) + len(diff.modified),
            added_count=len(diff.added),
            removed_count=len(diff.removed),
            modified_count=len(diff.modified),
            issues_created=issues_created,
        )

    def _find_item_by_hash(self, content_hash: str) -> ItemRecord | None:
        cur = self._index._db.execute(
            "SELECT * FROM memory_items WHERE content_hash=? AND removed_at IS NULL "
            "ORDER BY created_at LIMIT 1",
            (content_hash,),
        )
        row = cur.fetchone()
        if not row:
            return None
        from nanobot.agent.memory_index import _row_to_item
        return _row_to_item(row)

    async def _review_impact(
        self, item_id: str, trigger_ref: str | None, now: int,
    ) -> int:
        """Compute impact candidates, LLM-review, emit issues. Returns count."""
        impact = self._index.query_impact("item", item_id, depth=2)
        candidates: list[dict[str, Any]] = []
        seen_cand: set[str] = {item_id}  # don't list the changed item as its own dependent

        # Direct incoming edges (items that explicitly reference the changed one)
        for e in impact.incoming_edges:
            if e.from_kind != "item" or e.from_id in seen_cand:
                continue
            it = self._index.get_item(e.from_id)
            if not it:
                continue
            seen_cand.add(e.from_id)
            candidates.append({
                "id": it.id, "section": it.section_path,
                "content": it.content[:300],
                "source": "relation",
            })

        # Concept siblings (items sharing any concept with the changed item)
        concept_ids = self._index.list_concepts_for_item(item_id)
        for c_id in concept_ids:
            for sib_id in self._index.list_items_for_concept(c_id):
                if sib_id in seen_cand:
                    continue
                sib = self._index.get_item(sib_id)
                if not sib or sib.removed_at is not None:
                    continue
                seen_cand.add(sib_id)
                candidates.append({
                    "id": sib.id, "section": sib.section_path,
                    "content": sib.content[:300],
                    "source": "concept_sibling",
                })

        if not candidates:
            return 0

        changed = self._index.get_item(item_id)
        if not changed:
            return 0

        prompt = render_template(
            "agent/reconcile_impact.md",
            changed_item=json.dumps({
                "id": changed.id, "section": changed.section_path,
                "content": changed.content,
            }, ensure_ascii=False),
            candidates=json.dumps(candidates, ensure_ascii=False),
        )
        raw = await self._llm.complete(prompt)
        verdicts = _parse_json_array(raw)

        count = 0
        for v in verdicts:
            if not v.get("relevant"):
                continue
            cand_id = v.get("candidate_id")
            if not cand_id or cand_id not in seen_cand:
                continue
            self._index.add_issue(ConsistencyIssue(
                id="", trigger_event="dream_scan", trigger_ref=trigger_ref,
                issue_type="impact_unreviewed",
                subject_ids=json.dumps(
                    [{"kind": "item", "id": item_id},
                     {"kind": "item", "id": cand_id}],
                    ensure_ascii=False),
                description=v.get("action_hint", "受影响条目需复核"),
                severity=v.get("severity", "medium"),
                status="open", resolution=None,
                created_at=now, resolved_at=None,
            ))
            count += 1
        return count

    # -- watermark helpers ---------------------------------------------------

    def index_last_reconciled_commit(self) -> str | None:
        return self._index.get_meta("last_reconciled_commit")

    def set_last_reconciled_commit(self, sha: str) -> None:
        self._index.set_meta("last_reconciled_commit", sha)
