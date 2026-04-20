"""Tests for MemoryStore.backfill_embeddings."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nanobot.agent.memory import MemoryStore


def _fake_embedding_service(dim: int = 4):
    svc = type("FakeEmbed", (), {})()
    svc.dimensions = dim
    svc.embed_batch = AsyncMock(
        side_effect=lambda texts: [[0.1] * dim for _ in texts]
    )
    return svc


class TestBackfillEmbeddings:
    @pytest.mark.asyncio
    async def test_backfill_indexes_missing_entries(self, tmp_path):
        store = MemoryStore(tmp_path, embedding_dimensions=4)
        store.append_history("entry one")
        store.append_history("entry two")
        store.append_history("entry three")

        svc = _fake_embedding_service()
        count = await store.backfill_embeddings(svc, batch_size=10)
        assert count == 3

        # Verify vectors exist
        db = store._get_db()
        n = db.execute("SELECT count(*) FROM history_vec").fetchone()[0]
        assert n == 3

    @pytest.mark.asyncio
    async def test_backfill_idempotent(self, tmp_path):
        store = MemoryStore(tmp_path, embedding_dimensions=4)
        store.append_history("entry one")
        svc = _fake_embedding_service()

        count1 = await store.backfill_embeddings(svc)
        count2 = await store.backfill_embeddings(svc)
        assert count1 == 1
        assert count2 == 0

    @pytest.mark.asyncio
    async def test_backfill_respects_batch_size(self, tmp_path):
        store = MemoryStore(tmp_path, embedding_dimensions=4)
        for i in range(5):
            store.append_history(f"entry {i}")
        svc = _fake_embedding_service()

        count = await store.backfill_embeddings(svc, batch_size=2)
        assert count == 2  # Only first batch processed per call

    @pytest.mark.asyncio
    async def test_backfill_noop_when_vec_unavailable(self, tmp_path):
        store = MemoryStore(tmp_path)  # no embedding_dimensions
        store.append_history("entry one")
        svc = _fake_embedding_service()

        count = await store.backfill_embeddings(svc)
        assert count == 0
        svc.embed_batch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_backfill_handles_embedding_failure(self, tmp_path):
        store = MemoryStore(tmp_path, embedding_dimensions=4)
        store.append_history("e1")
        store.append_history("e2")
        svc = _fake_embedding_service()
        svc.embed_batch = AsyncMock(side_effect=RuntimeError("api down"))

        # Should not raise
        count = await store.backfill_embeddings(svc)
        assert count == 0
