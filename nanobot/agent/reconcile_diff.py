"""Diff algorithm: align old Chunks to new Chunks across MEMORY.md edits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from nanobot.agent.md_chunker import Chunk
from nanobot.agent.memory_index import _cosine, _unpack_floats


@dataclass
class ModifiedPair:
    old: Chunk
    new: Chunk
    similarity: float


@dataclass
class AmbiguousGroup:
    olds: list[Chunk]
    news: list[Chunk]
    kind: str  # 'one_to_many' | 'many_to_one'


@dataclass
class DiffResult:
    unchanged: list[Chunk] = field(default_factory=list)
    added: list[Chunk] = field(default_factory=list)
    removed: list[Chunk] = field(default_factory=list)
    modified: list[ModifiedPair] = field(default_factory=list)
    ambiguous: list[AmbiguousGroup] = field(default_factory=list)


def _default_cosine(a: bytes, b: bytes) -> float:
    va = _unpack_floats(a)
    vb = _unpack_floats(b)
    if len(va) != len(vb):
        return 0.0
    return _cosine(va, vb)


def align_items(
    old: list[Chunk],
    new: list[Chunk],
    old_embeddings: dict[str, bytes],
    new_embeddings: dict[str, bytes],
    threshold: float = 0.92,
    cosine_fn: Callable[[bytes, bytes], float] | None = None,
) -> DiffResult:
    """Pair old chunks to new chunks via hash-first, embedding-second matching."""
    result = DiffResult()
    cos = cosine_fn or _default_cosine

    # First pass: exact hash match
    old_by_hash: dict[str, list[int]] = {}
    for idx, ch in enumerate(old):
        old_by_hash.setdefault(ch.content_hash, []).append(idx)

    matched_old: set[int] = set()
    matched_new: set[int] = set()
    for j, ch in enumerate(new):
        candidates = old_by_hash.get(ch.content_hash, [])
        for i in candidates:
            if i in matched_old:
                continue
            result.unchanged.append(ch)
            matched_old.add(i)
            matched_new.add(j)
            break

    # Second pass: embedding similarity for remaining
    remaining_old = [i for i in range(len(old)) if i not in matched_old]
    remaining_new = [j for j in range(len(new)) if j not in matched_new]

    # Build similarity matrix only for remaining
    pairs: list[tuple[int, int, float]] = []
    for i in remaining_old:
        ie = old_embeddings.get(old[i].content_hash)
        if ie is None:
            continue
        for j in remaining_new:
            je = new_embeddings.get(new[j].content_hash)
            if je is None:
                continue
            score = cos(ie, je)
            if score >= threshold:
                pairs.append((i, j, score))

    # Greedy 1-to-1 match by descending score
    pairs.sort(key=lambda p: -p[2])
    old_to_new: dict[int, list[int]] = {}
    new_to_old: dict[int, list[int]] = {}
    for i, j, s in pairs:
        old_to_new.setdefault(i, []).append(j)
        new_to_old.setdefault(j, []).append(i)

    # Detect ambiguity: any old mapping to >1 new, or new mapping to >1 old
    ambiguous_olds: set[int] = set()
    ambiguous_news: set[int] = set()
    for i, js in old_to_new.items():
        if len(js) > 1:
            ambiguous_olds.add(i)
            ambiguous_news.update(js)
    for j, is_ in new_to_old.items():
        if len(is_) > 1:
            ambiguous_news.add(j)
            ambiguous_olds.update(is_)

    if ambiguous_olds or ambiguous_news:
        result.ambiguous.append(AmbiguousGroup(
            olds=[old[i] for i in sorted(ambiguous_olds)],
            news=[new[j] for j in sorted(ambiguous_news)],
            kind="one_to_many" if any(len(js) > 1 for js in old_to_new.values())
                 else "many_to_one",
        ))

    # Match unambiguous pairs (greedy by score)
    old_used: set[int] = set()
    new_used: set[int] = set()
    for i, j, s in pairs:
        if i in ambiguous_olds or j in ambiguous_news:
            continue
        if i in old_used or j in new_used:
            continue
        old_used.add(i)
        new_used.add(j)
        result.modified.append(ModifiedPair(old=old[i], new=new[j], similarity=s))

    # Leftovers
    for i in remaining_old:
        if i in old_used or i in ambiguous_olds:
            continue
        result.removed.append(old[i])
    for j in remaining_new:
        if j in new_used or j in ambiguous_news:
            continue
        result.added.append(new[j])

    return result
