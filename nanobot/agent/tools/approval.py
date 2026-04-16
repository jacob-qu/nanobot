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


class ApprovalEngine:
    """Manages dangerous command approval state and blocking wait."""

    def __init__(
        self,
        send_callback: Any,
        timeout: int = 300,
        allowlist: list[str] | None = None,
    ) -> None:
        self._send_callback = send_callback
        self._timeout = timeout
        self._allowlist: set[str] = set(allowlist or [])
        self._session_approved: dict[str, set[str]] = {}
        self._pending: dict[str, asyncio.Event] = {}
        self._results: dict[str, str] = {}

    async def check(
        self,
        command: str,
        session_key: str,
        channel: str,
        chat_id: str,
    ) -> tuple[bool, str]:
        """Check command and request approval if dangerous.

        Returns (approved, message). Blocks until user responds or timeout.
        """
        pattern_key = detect_dangerous_command(command)
        if pattern_key is None:
            return True, ""

        if pattern_key in self._allowlist:
            return True, ""

        session_set = self._session_approved.get(session_key, set())
        if pattern_key in session_set:
            return True, ""

        request_id = uuid.uuid4().hex[:12]
        event = asyncio.Event()
        self._pending[request_id] = event

        try:
            await self._send_callback(OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=(
                    f"⚠️ 危险命令需要审批\n\n"
                    f"```\n{command}\n```\n\n"
                    f"匹配规则：{pattern_key}"
                ),
                metadata={
                    "_approval_request": True,
                    "_approval_id": request_id,
                    "_approval_command": command,
                    "_approval_pattern": pattern_key,
                },
            ))
        except Exception:
            logger.exception("Failed to send approval request")
            self._pending.pop(request_id, None)
            return False, "failed to send approval request"

        try:
            await asyncio.wait_for(event.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            self._results.pop(request_id, None)
            return False, "approval timed out"

        decision = self._results.pop(request_id, "deny")
        self._pending.pop(request_id, None)

        if decision == "deny":
            return False, "denied by user"

        if decision == "session":
            self._session_approved.setdefault(session_key, set()).add(pattern_key)

        return True, ""

    def resolve(self, request_id: str, decision: str) -> bool:
        """Resolve a pending approval request. Returns True if found."""
        event = self._pending.get(request_id)
        if event is None:
            return False
        self._results[request_id] = decision
        event.set()
        return True

    def get_pending_requests(self) -> list[str]:
        """Return list of pending request IDs (for testing)."""
        return list(self._pending.keys())
