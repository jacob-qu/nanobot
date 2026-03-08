"""Context pruning: token-aware trimming of LLM context.

Inspired by OpenClaw's context management, this module prunes conversation
history to fit within a token budget while preserving the most valuable context.

Strategy (applied from oldest to newest, outside the protected zone):
  Phase 1 – Soft-trim old tool results (keep head + tail)
  Phase 2 – Hard-clear old tool results (replace with placeholder)
  Phase 3 – Drop entire old messages
"""

from __future__ import annotations

import copy

from loguru import logger


def estimate_tokens(text: str | None) -> int:
    """Rough token estimate: ~4 chars per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(msg: dict) -> int:
    """Estimate tokens for a single message (content + tool calls + overhead)."""
    tokens = 4  # per-message overhead (role, separators)
    content = msg.get("content")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    tokens += estimate_tokens(part.get("text", ""))
                elif part.get("type") == "image_url":
                    tokens += 300  # rough estimate for vision tokens
    if tool_calls := msg.get("tool_calls"):
        for tc in tool_calls:
            fn = tc.get("function", {})
            tokens += estimate_tokens(fn.get("name", ""))
            tokens += estimate_tokens(fn.get("arguments", ""))
    return tokens


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens for a message list."""
    return sum(estimate_message_tokens(m) for m in messages)


def _soft_trim(content: str, max_chars: int = 200, head: int = 80, tail: int = 80) -> str:
    """Keep head + tail of content, replace middle with a size note."""
    if len(content) <= max_chars:
        return content
    return f"{content[:head]}\n...[trimmed {len(content)} chars]...\n{content[-tail:]}"


def prune_context(
    messages: list[dict],
    max_tokens: int,
    keep_recent_turns: int = 3,
) -> tuple[list[dict], int]:
    """Prune messages to fit within a token budget.

    Returns ``(pruned_messages, estimated_tokens)``.

    Protected zone: system prompt (index 0) + the last *keep_recent_turns*
    user turns and everything that follows them.
    """
    total = estimate_messages_tokens(messages)
    if total <= max_tokens:
        return messages, total

    logger.info("Context pruning: ~{} tokens exceeds {} budget", total, max_tokens)
    msgs = copy.deepcopy(messages)

    # --- Locate the protected boundary ---
    # Walk backwards counting user messages.  Everything from the Nth user
    # message to the end is protected.
    user_count = 0
    protected_start = len(msgs)
    for i in range(len(msgs) - 1, 0, -1):
        if msgs[i].get("role") == "user":
            user_count += 1
            protected_start = i
            if user_count >= keep_recent_turns:
                break

    # --- Phase 1: soft-trim old tool results ---
    for i in range(1, protected_start):
        if msgs[i].get("role") == "tool" and isinstance(msgs[i].get("content"), str):
            msgs[i]["content"] = _soft_trim(msgs[i]["content"])

    total = estimate_messages_tokens(msgs)
    if total <= max_tokens:
        logger.info("Context pruned (soft-trim): ~{} tokens, {} messages", total, len(msgs))
        return msgs, total

    # --- Phase 2: hard-clear old tool results ---
    for i in range(1, protected_start):
        if msgs[i].get("role") == "tool":
            msgs[i]["content"] = "[pruned]"

    total = estimate_messages_tokens(msgs)
    if total <= max_tokens:
        logger.info("Context pruned (hard-clear): ~{} tokens, {} messages", total, len(msgs))
        return msgs, total

    # --- Phase 3: drop oldest messages one by one ---
    while protected_start > 1 and estimate_messages_tokens(msgs) > max_tokens:
        msgs.pop(1)
        protected_start -= 1

    # Ensure the first message after the system prompt is a user message
    # to avoid orphaned assistant/tool messages.
    while len(msgs) > 1 and msgs[1].get("role") not in ("user",):
        msgs.pop(1)

    total = estimate_messages_tokens(msgs)
    logger.info("Context pruned (drop): ~{} tokens, {} messages", total, len(msgs))
    return msgs, total
