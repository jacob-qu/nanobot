"""EmbeddingService: thin wrapper over OpenAI-compatible embeddings API.

Used by MemoryStore for semantic search over conversation history.
Reuses the existing `openai` SDK (already a core dependency); does not
modify the LLMProvider base class because embedding is a memory-layer
concern, not part of the chat provider contract.
"""

from __future__ import annotations

from openai import AsyncOpenAI


class EmbeddingService:
    """Generate embeddings via an OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        api_base: str | None,
        model: str,
        dimensions: int,
    ):
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector."""
        resp = await self._client.embeddings.create(
            input=text, model=self._model, dimensions=self._dimensions,
        )
        return list(resp.data[0].embedding)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single API call."""
        resp = await self._client.embeddings.create(
            input=texts, model=self._model, dimensions=self._dimensions,
        )
        return [list(d.embedding) for d in resp.data]
