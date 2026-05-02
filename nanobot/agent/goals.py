"""Persistent cross-turn goals — the Ralph loop for nanobot.

A goal is a standing user objective that survives across turns. After every
turn completes, a small judge model call asks "did the last response satisfy
the goal?". If not, nanobot re-injects a continuation prompt back through
the message bus and keeps working until the goal is done, the turn budget
is exhausted, the user pauses/clears it, or a real user message interrupts
the loop.

Design invariants:
  * State lives in ``Session.metadata["_goal"]`` — piggy-backs on the
    existing JSONL persistence, no new tables.
  * Judge failures are fail-OPEN (verdict="continue"). A broken judge
    never wedges progress; the turn budget is the backstop.
  * Continuation is delivered as a normal user-role ``InboundMessage`` via
    the bus, with ``metadata["_goal_continuation"]=True`` so callers can
    distinguish it from real user input. FIFO ordering of the bus inbound
    queue guarantees user messages preempt queued continuations.
  * No system-prompt mutation, no toolset swap — prompt caching stays intact.

Ported from Hermes' ``hermes_cli/goals.py`` (Eric Traut's original Codex CLI
``/goal`` design), simplified for nanobot's smaller surface.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session

DEFAULT_MAX_TURNS = 20
DEFAULT_JUDGE_TIMEOUT = 30.0
_JUDGE_RESPONSE_SNIPPET_CHARS = 4000
_META_KEY = "_goal"

CONTINUATION_PROMPT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Continue working toward this goal. Take the next concrete step. "
    "If you believe the goal is complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly and stop."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. You receive the goal text and the "
    "agent's most recent response. Your only job is to decide whether "
    "the goal is fully satisfied based on that response.\n\n"
    "A goal is DONE only when:\n"
    "- The response explicitly confirms the goal was completed, OR\n"
    "- The response clearly shows the final deliverable was produced, OR\n"
    "- The response explains the goal is unachievable / blocked / needs "
    "user input (treat this as DONE with reason describing the block).\n\n"
    "Otherwise the goal is NOT done — CONTINUE.\n\n"
    "Reply ONLY with a single JSON object on one line:\n"
    '{"done": <true|false>, "reason": "<one-sentence rationale>"}'
)

JUDGE_USER_PROMPT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "Is the goal satisfied?"
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class GoalState:
    """Serializable goal state stored in ``Session.metadata["_goal"]``."""

    goal: str
    status: str = "active"          # active | paused | done | cleared
    turns_used: int = 0
    max_turns: int = DEFAULT_MAX_TURNS
    created_at: float = 0.0
    last_turn_at: float = 0.0
    last_verdict: str | None = None
    last_reason: str | None = None
    paused_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoalState:
        return cls(
            goal=str(data.get("goal", "")),
            status=str(data.get("status", "active")),
            turns_used=int(data.get("turns_used", 0) or 0),
            max_turns=int(data.get("max_turns", DEFAULT_MAX_TURNS) or DEFAULT_MAX_TURNS),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_turn_at=float(data.get("last_turn_at", 0.0) or 0.0),
            last_verdict=data.get("last_verdict"),
            last_reason=data.get("last_reason"),
            paused_reason=data.get("paused_reason"),
        )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_judge_response(raw: str) -> tuple[bool, str]:
    """Parse judge JSON. Fail-open to ``(False, <reason>)`` on any error."""
    if not raw:
        return False, "judge returned empty response"

    text = raw.strip()
    # Strip markdown code fences the model may wrap JSON in.
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]

    data: dict[str, Any] | None = None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                data = None

    if not isinstance(data, dict):
        return False, f"judge reply was not JSON: {_truncate(raw, 200)!r}"

    done_val = data.get("done")
    if isinstance(done_val, str):
        done = done_val.strip().lower() in ("true", "yes", "1", "done")
    else:
        done = bool(done_val)
    reason = str(data.get("reason") or "").strip() or "no reason provided"
    return done, reason


async def judge_goal(
    provider: LLMProvider,
    model: str,
    goal: str,
    last_response: str,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
) -> tuple[str, str]:
    """Ask the LLM whether the goal is satisfied. Fail-open to ``continue``.

    Returns ``(verdict, reason)`` where verdict is ``"done"``, ``"continue"``,
    or ``"skipped"`` (empty goal only; every other failure returns
    ``"continue"`` so a broken judge never wedges progress).
    """
    if not goal.strip():
        return "skipped", "empty goal"
    if not last_response.strip():
        return "continue", "empty response (nothing to evaluate)"

    prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        goal=_truncate(goal, 2000),
        response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
    )

    try:
        resp = await asyncio.wait_for(
            provider.chat_with_retry(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
                temperature=0,
                max_tokens=200,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.info("goal judge: timeout — continuing")
        return "continue", "judge timeout"
    except Exception as exc:
        logger.info("goal judge: API call failed ({}) — continuing", exc)
        return "continue", f"judge error: {type(exc).__name__}"

    raw = ""
    try:
        # LLMResponse dataclass — .content is the text payload.
        raw = str(getattr(resp, "content", "") or "")
    except Exception:
        raw = ""

    done, reason = _parse_judge_response(raw)
    verdict = "done" if done else "continue"
    logger.info("goal judge: verdict={} reason={}", verdict, _truncate(reason, 120))
    return verdict, reason


# ---------------------------------------------------------------------------
# GoalManager — operates directly on Session.metadata["_goal"]
# ---------------------------------------------------------------------------

class GoalManager:
    """Per-session goal state backed by ``Session.metadata["_goal"]``.

    Construct per-call (cheap) — it reads from ``session.metadata`` on init
    and writes back on every mutation. The caller is responsible for
    ``sessions.save(session)`` afterward; nanobot already saves after every
    turn so this is usually automatic.
    """

    def __init__(self, session: Session, *, default_max_turns: int = DEFAULT_MAX_TURNS):
        self._session = session
        self._default_max_turns = int(default_max_turns or DEFAULT_MAX_TURNS)
        raw = session.metadata.get(_META_KEY)
        self._state: GoalState | None = None
        if isinstance(raw, dict):
            try:
                self._state = GoalState.from_dict(raw)
            except (TypeError, ValueError) as e:
                logger.warning("GoalManager: could not parse stored goal: {}", e)

    # --- introspection ----------------------------------------------------

    @property
    def state(self) -> GoalState | None:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_goal(self) -> bool:
        return self._state is not None and self._state.status in ("active", "paused")

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status == "cleared":
            return "No active goal. Set one with /goal <text>."
        turns = f"{s.turns_used}/{s.max_turns} turns"
        if s.status == "active":
            return f"⊙ Goal (active, {turns}): {s.goal}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Goal (paused, {turns}{extra}): {s.goal}"
        if s.status == "done":
            return f"✓ Goal done ({turns}): {s.goal}"
        return f"Goal ({s.status}, {turns}): {s.goal}"

    # --- mutation ---------------------------------------------------------

    def _persist(self) -> None:
        """Write current state into session.metadata (caller saves session)."""
        if self._state is None:
            self._session.metadata.pop(_META_KEY, None)
        else:
            self._session.metadata[_META_KEY] = self._state.to_dict()

    def set(self, goal: str, *, max_turns: int | None = None) -> GoalState:
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("goal text is empty")
        state = GoalState(
            goal=goal,
            status="active",
            turns_used=0,
            max_turns=int(max_turns) if max_turns else self._default_max_turns,
            created_at=time.time(),
            last_turn_at=0.0,
        )
        self._state = state
        self._persist()
        return state

    def pause(self, reason: str = "user-paused") -> GoalState | None:
        if not self._state:
            return None
        self._state.status = "paused"
        self._state.paused_reason = reason
        self._persist()
        return self._state

    def resume(self, *, reset_budget: bool = True) -> GoalState | None:
        if not self._state:
            return None
        self._state.status = "active"
        self._state.paused_reason = None
        if reset_budget:
            self._state.turns_used = 0
        self._persist()
        return self._state

    def clear(self) -> None:
        if self._state is None:
            return
        self._state.status = "cleared"
        self._persist()
        self._state = None
        # Remove from metadata entirely — cleared goals don't need to linger.
        self._session.metadata.pop(_META_KEY, None)

    def mark_done(self, reason: str) -> None:
        if not self._state:
            return
        self._state.status = "done"
        self._state.last_verdict = "done"
        self._state.last_reason = reason
        self._persist()

    # --- main entry point after every turn --------------------------------

    async def evaluate_after_turn(
        self,
        last_response: str,
        *,
        provider: LLMProvider,
        model: str,
    ) -> dict[str, Any]:
        """Run the judge, update state, return a decision dict.

        Decision keys:
          * ``status``: current goal status after update
          * ``should_continue``: bool — caller should enqueue a continuation
          * ``continuation_prompt``: str or None
          * ``verdict``: "done" | "continue" | "skipped" | "inactive"
          * ``reason``: str
          * ``message``: user-visible one-liner
        """
        state = self._state
        if state is None or state.status != "active":
            return {
                "status": state.status if state else None,
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "inactive",
                "reason": "no active goal",
                "message": "",
            }

        state.turns_used += 1
        state.last_turn_at = time.time()

        verdict, reason = await judge_goal(provider, model, state.goal, last_response)
        state.last_verdict = verdict
        state.last_reason = reason

        if verdict == "done":
            state.status = "done"
            self._persist()
            return {
                "status": "done",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "done",
                "reason": reason,
                "message": f"✓ Goal achieved: {reason}",
            }

        if state.turns_used >= state.max_turns:
            state.status = "paused"
            state.paused_reason = (
                f"turn budget exhausted ({state.turns_used}/{state.max_turns})"
            )
            self._persist()
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — {state.turns_used}/{state.max_turns} turns used. "
                    "Use /goal resume to keep going, or /goal clear to stop."
                ),
            }

        self._persist()
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": CONTINUATION_PROMPT_TEMPLATE.format(goal=state.goal),
            "verdict": "continue",
            "reason": reason,
            "message": (
                f"↻ Continuing toward goal ({state.turns_used}/{state.max_turns}): {reason}"
            ),
        }


__all__ = [
    "GoalState",
    "GoalManager",
    "CONTINUATION_PROMPT_TEMPLATE",
    "DEFAULT_MAX_TURNS",
    "judge_goal",
]
