"""Tests for context compression enhancements."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nanobot.agent.memory import Consolidator

_PRUNED_TOOL_PLACEHOLDER = Consolidator._PRUNED_TOOL_PLACEHOLDER


def _make_consolidator(context_window_tokens: int = 128_000) -> Consolidator:
    """Create a minimal Consolidator for unit testing."""
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 4096
    return Consolidator(
        store=MagicMock(),
        provider=provider,
        model="test-model",
        sessions=MagicMock(),
        context_window_tokens=context_window_tokens,
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
    )


class TestPruneOldToolResults:
    def test_prunes_large_tool_results(self):
        """Tool results >200 chars should be replaced with placeholder."""
        messages = [
            {"role": "user", "content": "run ls"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "bash", "arguments": '{"cmd":"ls"}'}}
            ]},
            {"role": "tool", "content": "x" * 300, "tool_call_id": "tc1"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "you're welcome"},
        ]
        c = _make_consolidator()
        pruned, count = c._prune_old_tool_results(messages, protect_tail_idx=4)
        assert count == 1
        assert pruned[2]["content"] == _PRUNED_TOOL_PLACEHOLDER

    def test_skips_small_tool_results(self):
        """Tool results <=200 chars should be left intact."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "short", "tool_call_id": "tc1"},
            {"role": "user", "content": "ok"},
        ]
        c = _make_consolidator()
        pruned, count = c._prune_old_tool_results(messages, protect_tail_idx=2)
        assert count == 0
        assert pruned[1]["content"] == "short"

    def test_protects_tail(self):
        """Tool results in the tail protection zone should not be pruned."""
        messages = [
            {"role": "user", "content": "old"},
            {"role": "tool", "content": "x" * 300, "tool_call_id": "tc1"},
            {"role": "user", "content": "recent"},
            {"role": "tool", "content": "y" * 300, "tool_call_id": "tc2"},
        ]
        c = _make_consolidator()
        pruned, count = c._prune_old_tool_results(messages, protect_tail_idx=2)
        assert count == 1
        assert pruned[1]["content"] == _PRUNED_TOOL_PLACEHOLDER
        assert pruned[3]["content"] == "y" * 300  # tail protected

    def test_does_not_modify_non_tool_messages(self):
        """User and assistant messages should never be pruned."""
        messages = [
            {"role": "user", "content": "x" * 500},
            {"role": "assistant", "content": "y" * 500},
        ]
        c = _make_consolidator()
        pruned, count = c._prune_old_tool_results(messages, protect_tail_idx=1)
        assert count == 0
        assert pruned[0]["content"] == "x" * 500

    def test_already_pruned_skipped(self):
        """Already-pruned tool results should not be counted again."""
        messages = [
            {"role": "tool", "content": _PRUNED_TOOL_PLACEHOLDER, "tool_call_id": "tc1"},
            {"role": "user", "content": "ok"},
        ]
        c = _make_consolidator()
        pruned, count = c._prune_old_tool_results(messages, protect_tail_idx=1)
        assert count == 0


class TestStructuredSummary:
    @pytest.mark.asyncio
    async def test_first_archive_uses_structured_template(self):
        """First archive should use the structured 'from scratch' prompt."""
        c = _make_consolidator()
        calls = []

        async def _capture_call(**kwargs):
            calls.append(kwargs)
            return MagicMock(content="## Goal\nTest goal\n## Progress\n### Done\nNothing")

        c.provider.chat_with_retry = _capture_call
        c.store.append_history = MagicMock()

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = await c.archive(messages, session_key="cli:test")
        assert result is not None
        assert "## Goal" in calls[0]["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_subsequent_archive_uses_iterative_update(self):
        """Second archive for same session should include previous summary."""
        c = _make_consolidator()
        c._previous_summary["cli:test"] = "## Goal\nPrevious goal"
        calls = []

        async def _capture_call(**kwargs):
            calls.append(kwargs)
            return MagicMock(content="## Goal\nUpdated goal")

        c.provider.chat_with_retry = _capture_call
        c.store.append_history = MagicMock()

        messages = [{"role": "user", "content": "new msg"}]
        result = await c.archive(messages, session_key="cli:test")
        assert result is not None
        prompt_text = calls[0]["messages"][0]["content"]
        assert "Previous goal" in prompt_text
        assert "PREVIOUS SUMMARY" in prompt_text

    @pytest.mark.asyncio
    async def test_summary_stored_for_iterative_use(self):
        """After archive, summary should be stored in _previous_summary."""
        c = _make_consolidator()

        async def _fake_llm(**kwargs):
            return MagicMock(content="## Goal\nStored summary")

        c.provider.chat_with_retry = _fake_llm
        c.store.append_history = MagicMock()

        await c.archive(
            [{"role": "user", "content": "hi"}],
            session_key="cli:test",
        )
        assert "cli:test" in c._previous_summary
        assert "Stored summary" in c._previous_summary["cli:test"]

    @pytest.mark.asyncio
    async def test_summary_has_compaction_prefix(self):
        """Archive result should start with the compaction prefix."""
        c = _make_consolidator()

        async def _fake_llm(**kwargs):
            return MagicMock(content="## Goal\nTest")

        c.provider.chat_with_retry = _fake_llm
        c.store.append_history = MagicMock()

        result = await c.archive(
            [{"role": "user", "content": "hi"}],
            session_key="cli:test",
        )
        assert result.startswith("[CONTEXT COMPACTION]")


class TestSummaryFailureCooldown:
    @pytest.mark.asyncio
    async def test_cooldown_after_failure(self):
        """After LLM failure, subsequent calls within cooldown should skip LLM."""
        c = _make_consolidator()
        call_count = 0

        async def _failing_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("API down")

        c.provider.chat_with_retry = _failing_llm
        c.store.raw_archive = MagicMock()

        await c.archive([{"role": "user", "content": "a"}], session_key="cli:test")
        assert call_count == 1

        # Second call should skip LLM (cooldown active)
        await c.archive([{"role": "user", "content": "b"}], session_key="cli:test")
        assert call_count == 1  # Not incremented

    @pytest.mark.asyncio
    async def test_cooldown_expires(self):
        """After cooldown expires, LLM should be called again."""
        c = _make_consolidator()
        call_count = 0

        async def _failing_then_ok(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API down")
            return MagicMock(content="## Goal\nRecovered")

        c.provider.chat_with_retry = _failing_then_ok
        c.store.raw_archive = MagicMock()
        c.store.append_history = MagicMock()

        await c.archive([{"role": "user", "content": "a"}], session_key="cli:test")
        assert call_count == 1

        # Manually expire the cooldown
        c._summary_failure_cooldown["cli:test"] = 0.0

        await c.archive([{"role": "user", "content": "b"}], session_key="cli:test")
        assert call_count == 2
