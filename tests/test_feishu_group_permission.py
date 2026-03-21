"""Tests for Feishu group safety features (permissions, context buffer, guest mode)."""
from nanobot.channels.feishu import FeishuConfig


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
