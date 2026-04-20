"""Tests for EmbeddingService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.embedding import EmbeddingService


def _make_service(embedding_vectors: list[list[float]]) -> EmbeddingService:
    """Build an EmbeddingService with a mocked AsyncOpenAI client."""
    service = EmbeddingService(
        api_key="test-key",
        api_base=None,
        model="text-embedding-3-small",
        dimensions=len(embedding_vectors[0]),
    )
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=v) for v in embedding_vectors]
    service._client.embeddings.create = AsyncMock(return_value=fake_response)
    return service


class TestEmbeddingService:
    def test_dimensions_property(self):
        service = EmbeddingService(
            api_key="k", api_base=None, model="m", dimensions=4,
        )
        assert service.dimensions == 4

    @pytest.mark.asyncio
    async def test_embed_returns_single_vector(self):
        service = _make_service([[0.1, 0.2, 0.3, 0.4]])
        vec = await service.embed("hello")
        assert vec == [0.1, 0.2, 0.3, 0.4]
        service._client.embeddings.create.assert_awaited_once_with(
            input="hello", model="text-embedding-3-small", dimensions=4,
        )

    @pytest.mark.asyncio
    async def test_embed_batch_returns_multiple_vectors(self):
        service = _make_service([[0.1, 0.2], [0.3, 0.4]])
        vecs = await service.embed_batch(["a", "b"])
        assert vecs == [[0.1, 0.2], [0.3, 0.4]]
        service._client.embeddings.create.assert_awaited_once_with(
            input=["a", "b"], model="text-embedding-3-small", dimensions=2,
        )

    @pytest.mark.asyncio
    async def test_embed_propagates_client_error(self):
        service = _make_service([[0.0]])
        service._client.embeddings.create = AsyncMock(
            side_effect=RuntimeError("network down"),
        )
        with pytest.raises(RuntimeError, match="network down"):
            await service.embed("hello")
