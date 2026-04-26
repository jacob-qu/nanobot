"""Tests for reconcile_diff — aligns old/new item sets across MD edits."""

from nanobot.agent.md_chunker import Chunk
from nanobot.agent.reconcile_diff import DiffResult, align_items


def _ch(content: str, section: str = "A", item_type: str = "list_item") -> Chunk:
    import hashlib
    return Chunk(
        item_type=item_type, content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        section_path=section,
    )


class TestHashMatchingFirstPass:
    def test_identical_chunks_are_unchanged(self):
        old = [_ch("foo"), _ch("bar")]
        new = [_ch("foo"), _ch("bar")]
        result = align_items(old, new, old_embeddings={}, new_embeddings={})
        assert len(result.unchanged) == 2
        assert not result.added and not result.removed and not result.modified

    def test_reordered_same_content_still_unchanged(self):
        old = [_ch("foo"), _ch("bar")]
        new = [_ch("bar"), _ch("foo")]
        result = align_items(old, new, old_embeddings={}, new_embeddings={})
        assert len(result.unchanged) == 2


class TestEmbeddingSecondPass:
    def test_similar_new_maps_to_old_as_modified(self):
        old_ch = _ch("三档待办判断：A/B/C")
        new_ch = _ch("三档待办判断：A/B/C/D")  # new rule added
        emb_a = bytes([1, 2, 3] * 4)
        emb_b = bytes([1, 2, 3] * 4)

        def _mock_cosine(a, b):
            return 0.95 if a == b else 0.1

        result = align_items(
            [old_ch], [new_ch],
            old_embeddings={old_ch.content_hash: emb_a},
            new_embeddings={new_ch.content_hash: emb_b},
            threshold=0.92,
            cosine_fn=_mock_cosine,
        )
        assert len(result.modified) == 1
        assert result.modified[0].old.content == "三档待办判断：A/B/C"
        assert result.modified[0].new.content == "三档待办判断：A/B/C/D"

    def test_unrelated_new_becomes_added(self):
        old_ch = _ch("foo")
        new_ch = _ch("totally different content")
        result = align_items(
            [old_ch], [new_ch],
            old_embeddings={}, new_embeddings={},
            threshold=0.92,
        )
        assert len(result.added) == 1 and len(result.removed) == 1


class TestAmbiguous:
    def test_one_old_to_two_new_flagged_ambiguous(self):
        old_ch = _ch("A and B")
        new_a = _ch("A")
        new_b = _ch("B")

        def _cosine(a, b):
            return 0.93 if a and b else 0.0

        result = align_items(
            [old_ch], [new_a, new_b],
            old_embeddings={old_ch.content_hash: b"x"},
            new_embeddings={new_a.content_hash: b"x", new_b.content_hash: b"x"},
            threshold=0.9,
            cosine_fn=_cosine,
        )
        assert len(result.ambiguous) >= 1
