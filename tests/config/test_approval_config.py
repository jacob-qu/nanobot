"""Tests for ApprovalConfig schema."""

from nanobot.config.schema import ApprovalConfig, ToolsConfig


def test_approval_config_defaults():
    cfg = ApprovalConfig()
    assert cfg.enabled is False
    assert cfg.timeout == 300
    assert cfg.allowlist == []


def test_approval_config_custom():
    cfg = ApprovalConfig(enabled=True, timeout=120, allowlist=["git_reset_hard"])
    assert cfg.enabled is True
    assert cfg.timeout == 120
    assert cfg.allowlist == ["git_reset_hard"]


def test_tools_config_has_approval():
    tc = ToolsConfig()
    assert hasattr(tc, "approval")
    assert isinstance(tc.approval, ApprovalConfig)
    assert tc.approval.enabled is False
