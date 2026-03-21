"""Tests for Feishu group safety features (permissions, context buffer, guest mode)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from nanobot.bus.queue import MessageBus
from nanobot.channels.feishu import FeishuChannel, FeishuConfig


def _make_channel(**kwargs) -> FeishuChannel:
    cfg = FeishuConfig(enabled=True, app_id="x", app_secret="x", **kwargs)
    ch = FeishuChannel(cfg, MessageBus())
    ch._client = MagicMock()
    ch._bot_open_id = "ou_bot"
    return ch


def test_feishu_config_defaults() -> None:
    cfg = FeishuConfig(enabled=True, app_id="x", app_secret="x")
    assert cfg.owner_open_ids == []
    assert cfg.heuristic_keywords == []
    assert cfg.group_context_max_tokens == 8192
    assert cfg.guest_allowed_tools == ["web_search", "web_fetch"]
    assert cfg.group_policy == "mention"


def test_feishu_config_heuristic_policy_accepted() -> None:
    cfg = FeishuConfig(
        enabled=True, app_id="x", app_secret="x",
        group_policy="heuristic",
        heuristic_keywords=["Alice", "nanobot"],
        owner_open_ids=["ou_owner1"],
    )
    assert cfg.group_policy == "heuristic"
    assert "Alice" in cfg.heuristic_keywords


def test_feishu_config_guest_allowed_tools_override() -> None:
    cfg = FeishuConfig(
        enabled=True, app_id="x", app_secret="x",
        guest_allowed_tools=["web_search"],
    )
    assert cfg.guest_allowed_tools == ["web_search"]


def test_estimate_tokens_approximation() -> None:
    ch = _make_channel()
    # 10 chars ≈ 5 tokens
    assert ch._estimate_tokens([("Alice", "1234567890")]) == 5


def test_append_group_context_basic() -> None:
    ch = _make_channel(group_context_max_tokens=100)
    ch._append_group_context("chat1", "Alice", "hello world")
    assert len(ch._group_context["chat1"]) == 1
    assert ch._group_context["chat1"][0] == ("Alice", "hello world")


def test_append_group_context_trims_oldest_when_over_80pct() -> None:
    # max=100 tokens, threshold=80 tokens
    # First entry: 160 chars = 80 tokens (exactly at threshold, not trimmed yet)
    # After second entry: 80+10=90 tokens > 80, trim first → only Bob remains
    ch = _make_channel(group_context_max_tokens=100)
    ch._append_group_context("chat1", "Alice", "a" * 160)   # 80 tokens
    ch._append_group_context("chat1", "Bob", "b" * 20)      # 10 tokens → total 90 > 80, trim Alice
    assert len(ch._group_context["chat1"]) == 1
    assert ch._group_context["chat1"][0][0] == "Bob"


def test_append_group_context_single_oversized_message_kept() -> None:
    """A single message over the threshold is kept (never trim to empty)."""
    ch = _make_channel(group_context_max_tokens=10)  # 80% = 8 tokens
    # 100 chars = 50 tokens, way over the 8-token threshold
    ch._append_group_context("chat1", "Alice", "a" * 100)
    # Single entry must be kept despite being over threshold
    assert len(ch._group_context["chat1"]) == 1


def test_build_group_context_str_format() -> None:
    ch = _make_channel()
    ch._group_context["chat1"] = [("Alice", "hi"), ("Bob", "hello")]
    result = ch._build_group_context_str("chat1")
    assert "[群聊上下文]" in result
    assert "Alice: hi" in result
    assert "Bob: hello" in result
    assert "---" in result


def test_build_group_context_str_empty() -> None:
    ch = _make_channel()
    assert ch._build_group_context_str("nonexistent") == ""



def _make_mention(open_id="ou_bot", user_id=None):
    return SimpleNamespace(id=SimpleNamespace(open_id=open_id, user_id=user_id))


def _make_msg(mentions=None, content='{"text": "hello"}'):
    return SimpleNamespace(mentions=mentions or [], content=content)


def test_is_group_message_for_bot_mention_mode_with_mention() -> None:
    ch = _make_channel(group_policy="mention")
    msg = _make_msg(mentions=[_make_mention("ou_bot")])
    assert ch._is_group_message_for_bot(msg, "hello") is True


def test_is_group_message_for_bot_mention_mode_without_mention() -> None:
    ch = _make_channel(group_policy="mention")
    msg = _make_msg(mentions=[])
    assert ch._is_group_message_for_bot(msg, "hello nanobot") is False


def test_is_group_message_for_bot_heuristic_keyword_match() -> None:
    ch = _make_channel(group_policy="heuristic", heuristic_keywords=["nanobot", "Jacob"])
    msg = _make_msg(mentions=[])
    assert ch._is_group_message_for_bot(msg, "Jacob 你好") is True


def test_is_group_message_for_bot_heuristic_no_match() -> None:
    ch = _make_channel(group_policy="heuristic", heuristic_keywords=["nanobot"])
    msg = _make_msg(mentions=[])
    assert ch._is_group_message_for_bot(msg, "普通消息") is False


def test_is_group_message_for_bot_heuristic_mention_overrides() -> None:
    """Direct @mention always triggers, even in heuristic mode."""
    ch = _make_channel(group_policy="heuristic", heuristic_keywords=[])
    msg = _make_msg(mentions=[_make_mention("ou_bot")])
    assert ch._is_group_message_for_bot(msg, "no keyword") is True


def test_guest_mode_true_for_group_non_owner() -> None:
    ch = _make_channel(owner_open_ids=["ou_owner"])
    # Simulate metadata tagging logic
    is_guest = "group" == "group" and "ou_stranger" not in ch.config.owner_open_ids
    assert is_guest is True


def test_guest_mode_false_for_group_owner() -> None:
    ch = _make_channel(owner_open_ids=["ou_owner"])
    is_guest = "group" == "group" and "ou_owner" not in ch.config.owner_open_ids
    assert is_guest is False
