"""Thin adapter from nanobot's LLMProvider to ReconcileEngine's _LLMLike protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


class ProviderLLMAdapter:
    """Wrap LLMProvider.chat_with_retry() behind a simple .complete(prompt) -> str."""

    def __init__(self, provider: "LLMProvider", model: str):
        self._provider = provider
        self._model = model

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        resp = await self._provider.chat_with_retry(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            tool_choice=None,
        )
        return resp.content or ""
