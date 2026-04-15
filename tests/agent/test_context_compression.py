"""Tests for context compression enhancements."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nanobot.agent.autocompact import AutoCompact
from nanobot.agent.memory import Consolidator
from nanobot.session.manager import Session

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


class TestBoundaryAlignment:
    def test_align_forward_skips_tool_messages(self):
        """Boundary should skip past consecutive tool messages at the start."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "result1", "tool_call_id": "tc1"},
            {"role": "tool", "content": "result2", "tool_call_id": "tc2"},
            {"role": "user", "content": "ok"},
        ]
        c = _make_consolidator()
        assert c._align_boundary_forward(messages, 1) == 3

    def test_align_forward_noop_on_non_tool(self):
        """If idx is already on a non-tool message, return it unchanged."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        c = _make_consolidator()
        assert c._align_boundary_forward(messages, 0) == 0

    def test_align_backward_pulls_before_assistant(self):
        """Boundary inside tool group should pull back to before the assistant."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
            {"role": "user", "content": "ok"},
        ]
        c = _make_consolidator()
        assert c._align_boundary_backward(messages, 2) == 1

    def test_align_backward_noop_when_no_tool_group(self):
        """If boundary is not in a tool group, return unchanged."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "ok"},
        ]
        c = _make_consolidator()
        assert c._align_boundary_backward(messages, 2) == 2


class TestSanitizeToolPairs:
    def test_removes_orphaned_tool_results(self):
        """Tool results without matching assistant tool_calls should be removed."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "orphan", "tool_call_id": "tc_gone"},
            {"role": "assistant", "content": "ok"},
        ]
        c = _make_consolidator()
        sanitized = c._sanitize_tool_pairs(messages)
        assert len(sanitized) == 2
        assert not any(m.get("tool_call_id") == "tc_gone" for m in sanitized)

    def test_inserts_stub_for_orphaned_calls(self):
        """Assistant tool_calls without results should get a stub result."""
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}
            ]},
            {"role": "user", "content": "ok"},
        ]
        c = _make_consolidator()
        sanitized = c._sanitize_tool_pairs(messages)
        assert len(sanitized) == 3
        stub = sanitized[1]
        assert stub["role"] == "tool"
        assert stub["tool_call_id"] == "tc1"
        assert "earlier conversation" in stub["content"]

    def test_matched_pairs_unchanged(self):
        """Complete tool_call/result pairs should be left intact."""
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "tc1", "function": {"name": "bash", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
            {"role": "user", "content": "thanks"},
        ]
        c = _make_consolidator()
        sanitized = c._sanitize_tool_pairs(messages)
        assert len(sanitized) == 3
        assert sanitized[1]["content"] == "result"


class TestConsolidatorTailProtection:
    def test_tail_protect_tokens_default(self):
        """Default tail_protect_tokens should be 20% of context_window_tokens."""
        c = _make_consolidator(context_window_tokens=100_000)
        assert c.tail_protect_tokens == 20_000

    def test_boundary_respects_tail_budget(self):
        """pick_consolidation_boundary should not eat into the tail budget."""
        c = _make_consolidator(context_window_tokens=100_000)
        # tail_protect_tokens = 20_000
        session = Session(key="test", messages=[], last_consolidated=0)
        # Each message ~10 tokens; 100 messages = ~1000 tokens total
        for i in range(100):
            session.messages.append({"role": "user" if i % 2 == 0 else "assistant",
                                     "content": f"msg {i}"})
        # Ask to remove 500 tokens — should succeed but protect tail
        result = c.pick_consolidation_boundary(session, tokens_to_remove=500)
        assert result is not None
        end_idx = result[0]
        # Should not consume all messages
        assert end_idx < len(session.messages) - 3


class TestAutoCompactTokenBudget:
    def test_token_budget_replaces_fixed_count(self):
        """AutoCompact with context_window_tokens should use token budget, not fixed 8."""
        ac = AutoCompact(
            sessions=MagicMock(),
            consolidator=MagicMock(),
            session_ttl_minutes=15,
            context_window_tokens=128_000,
        )
        assert ac._tail_token_budget > 0
        assert ac._RECENT_SUFFIX_MESSAGES == 3  # hard minimum

    def test_fallback_to_fixed_count_without_context_tokens(self):
        """AutoCompact without context_window_tokens should use fixed 8 messages."""
        ac = AutoCompact(
            sessions=MagicMock(),
            consolidator=MagicMock(),
            session_ttl_minutes=15,
        )
        assert ac._tail_token_budget == 0

    def test_split_unconsolidated_uses_token_budget(self):
        """With token budget, split should protect roughly the right amount."""
        sessions = MagicMock()
        # Use a small context_window so tail_token_budget = 20% of 1000 = 200 tokens
        ac = AutoCompact(
            sessions=sessions,
            consolidator=MagicMock(),
            session_ttl_minutes=15,
            context_window_tokens=1_000,
        )
        session = Session(key="test", messages=[], last_consolidated=0)
        for i in range(20):
            session.messages.append(
                {"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"message number {i} " + "x" * 100}
            )
        archive, kept = ac._split_unconsolidated(session)
        # Should keep more than the hard minimum of 3
        assert len(kept) >= ac._RECENT_SUFFIX_MESSAGES
        # Should archive something
        assert len(archive) > 0
