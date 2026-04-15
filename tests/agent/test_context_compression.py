"""Tests for context compression enhancements."""

from __future__ import annotations

from unittest.mock import MagicMock

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
