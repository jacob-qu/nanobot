"""Command approval system for dangerous shell commands.

Provides a pattern-based detection layer for medium-risk commands and an
async approval engine that blocks execution until user consent is received
via chat-platform buttons.
"""

from __future__ import annotations

import asyncio
import contextvars
import re
import unicodedata
import uuid
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage

# ---------------------------------------------------------------------------
# Dangerous command patterns (medium risk — user may legitimately need these)
# ---------------------------------------------------------------------------

APPROVAL_PATTERNS: list[tuple[str, str]] = [
    # (pattern_key, regex)
    ("recursive_chmod", r"\bchmod\s+-R\b"),
    ("recursive_chown", r"\bchown\s+-R\b"),
    ("git_reset_hard", r"\bgit\s+reset\s+--hard\b"),
    ("git_force_push", r"\bgit\s+push\b.*--force"),
    ("git_clean", r"\bgit\s+clean\s+-[fdx]"),
    ("git_branch_delete", r"\bgit\s+branch\s+-D\b"),
    ("drop_table", r"\bDROP\s+(TABLE|DATABASE)\b"),
    ("delete_no_where", r"\bDELETE\s+FROM\b(?!.*\bWHERE\b)"),
    ("truncate_table", r"\bTRUNCATE\b"),
    ("kill_process", r"\b(kill|pkill|killall)\b"),
    ("systemctl_stop", r"\bsystemctl\s+(stop|disable|mask)\b"),
    ("curl_pipe_sh", r"\bcurl\b.*\|\s*(bash|sh|zsh)"),
    ("wget_pipe_sh", r"\bwget\b.*\|\s*(bash|sh|zsh)"),
    ("docker_rm_force", r"\bdocker\s+(rm\s+-f|system\s+prune)"),
]

_COMPILED_PATTERNS: list[tuple[str, re.Pattern]] = [
    (key, re.compile(pattern, re.IGNORECASE | re.DOTALL))
    for key, pattern in APPROVAL_PATTERNS
]


def _normalize_command(command: str) -> str:
    """NFKC-normalize and strip ANSI escape sequences."""
    text = unicodedata.normalize("NFKC", command)
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)  # strip CSI sequences
    text = re.sub(r"\x1b\][^\x07]*\x07", "", text)       # strip OSC sequences
    text = text.replace("\x00", "")                        # strip null bytes
    return text


def detect_dangerous_command(command: str) -> str | None:
    """Return the pattern_key if *command* matches a dangerous pattern, else None."""
    normalized = _normalize_command(command)
    for key, pattern in _COMPILED_PATTERNS:
        if pattern.search(normalized):
            return key
    return None
