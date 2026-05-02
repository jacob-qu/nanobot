"""Tests for the /goal Ralph loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.goals import (
    DEFAULT_MAX_TURNS,
    GoalManager,
    GoalState,
    _parse_judge_response,
    judge_goal,
)
from nanobot.providers.base import LLMResponse
from nanobot.session.manager import Session


# ---------------------------------------------------------------------------
# _parse_judge_response — JSON parsing corner cases
# ---------------------------------------------------------------------------

class TestParseJudge:
    def test_plain_json_done(self):
        done, reason = _parse_judge_response('{"done": true, "reason": "ok"}')
        assert done is True
        assert reason == "ok"

    def test_plain_json_continue(self):
        done, reason = _parse_judge_response('{"done": false, "reason": "need more"}')
        assert done is False
        assert reason == "need more"

    def test_fenced_json(self):
        raw = '```json\n{"done": true, "reason": "done via fence"}\n```'
        done, reason = _parse_judge_response(raw)
        assert done is True
        assert reason == "done via fence"

    def test_json_embedded_in_prose(self):
        raw = 'Judge output: {"done": true, "reason": "great"} — thanks!'
        done, reason = _parse_judge_response(raw)
        assert done is True
        assert reason == "great"

    def test_empty_defaults_to_continue(self):
        done, reason = _parse_judge_response("")
        assert done is False
        assert "empty" in reason.lower()

    def test_non_json_defaults_to_continue(self):
        done, reason = _parse_judge_response("I think it's done!")
        assert done is False
        assert "not json" in reason.lower() or "was not" in reason.lower()

    def test_done_as_string(self):
        # Some models return "true" as a string
        done, reason = _parse_judge_response('{"done": "true", "reason": "x"}')
        assert done is True

    def test_missing_reason_has_default(self):
        done, reason = _parse_judge_response('{"done": true}')
        assert done is True
        assert reason  # non-empty fallback


# ---------------------------------------------------------------------------
# Fake provider for judge_goal
# ---------------------------------------------------------------------------

@dataclass
class _FakeProvider:
    """Minimal LLMProvider stand-in for judge tests."""

    content_to_return: str = ""
    should_raise: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def chat_with_retry(self, **kwargs):
        self.calls.append(kwargs)
        if self.should_raise is not None:
            raise self.should_raise
        return LLMResponse(content=self.content_to_return)


class TestJudgeGoal:
    @pytest.mark.asyncio
    async def test_empty_goal_skipped(self):
        provider = _FakeProvider()
        verdict, reason = await judge_goal(provider, "model", "", "some response")
        assert verdict == "skipped"
        assert not provider.calls

    @pytest.mark.asyncio
    async def test_empty_response_continues(self):
        provider = _FakeProvider()
        verdict, reason = await judge_goal(provider, "model", "fix bugs", "")
        assert verdict == "continue"
        assert not provider.calls  # didn't waste a call

    @pytest.mark.asyncio
    async def test_done_verdict(self):
        provider = _FakeProvider(content_to_return='{"done": true, "reason": "all tests pass"}')
        verdict, reason = await judge_goal(provider, "model", "fix tests", "tests now pass")
        assert verdict == "done"
        assert reason == "all tests pass"

    @pytest.mark.asyncio
    async def test_continue_verdict(self):
        provider = _FakeProvider(content_to_return='{"done": false, "reason": "still broken"}')
        verdict, reason = await judge_goal(provider, "model", "fix tests", "tried X")
        assert verdict == "continue"
        assert reason == "still broken"

    @pytest.mark.asyncio
    async def test_provider_failure_fail_open(self):
        provider = _FakeProvider(should_raise=RuntimeError("boom"))
        verdict, reason = await judge_goal(provider, "model", "fix tests", "progress")
        assert verdict == "continue"
        assert "error" in reason.lower()


# ---------------------------------------------------------------------------
# GoalManager — state transitions
# ---------------------------------------------------------------------------

def _make_session(tmp: Path) -> Session:
    return Session(key="cli:direct")


class TestGoalManager:
    def test_new_manager_no_state(self, tmp_path):
        mgr = GoalManager(_make_session(tmp_path))
        assert mgr.state is None
        assert not mgr.has_goal()
        assert not mgr.is_active()

    def test_set_goal_persists_to_metadata(self, tmp_path):
        session = _make_session(tmp_path)
        mgr = GoalManager(session)
        mgr.set("Fix every test")
        assert mgr.is_active()
        assert "_goal" in session.metadata
        assert session.metadata["_goal"]["goal"] == "Fix every test"

    def test_set_empty_raises(self, tmp_path):
        mgr = GoalManager(_make_session(tmp_path))
        with pytest.raises(ValueError):
            mgr.set("  ")

    def test_pause_and_resume(self, tmp_path):
        session = _make_session(tmp_path)
        mgr = GoalManager(session)
        mgr.set("X")
        mgr.pause("testing")
        assert mgr.state.status == "paused"
        assert mgr.state.paused_reason == "testing"
        mgr.resume()
        assert mgr.state.status == "active"
        assert mgr.state.paused_reason is None
        assert mgr.state.turns_used == 0  # resume resets budget by default

    def test_clear_wipes_metadata(self, tmp_path):
        session = _make_session(tmp_path)
        mgr = GoalManager(session)
        mgr.set("X")
        mgr.clear()
        assert mgr.state is None
        assert "_goal" not in session.metadata

    def test_second_instance_loads_state(self, tmp_path):
        session = _make_session(tmp_path)
        mgr1 = GoalManager(session)
        mgr1.set("Persistent goal")
        # New manager from same session — should pick up state
        mgr2 = GoalManager(session)
        assert mgr2.is_active()
        assert mgr2.state.goal == "Persistent goal"


class TestEvaluateAfterTurn:
    @pytest.mark.asyncio
    async def test_inactive_noop(self, tmp_path):
        mgr = GoalManager(_make_session(tmp_path))
        provider = _FakeProvider()
        decision = await mgr.evaluate_after_turn("resp", provider=provider, model="m")
        assert decision["verdict"] == "inactive"
        assert not decision["should_continue"]
        assert not provider.calls

    @pytest.mark.asyncio
    async def test_done_stops_loop(self, tmp_path):
        mgr = GoalManager(_make_session(tmp_path))
        mgr.set("X")
        provider = _FakeProvider(
            content_to_return='{"done": true, "reason": "ship it"}'
        )
        decision = await mgr.evaluate_after_turn(
            "the answer is 42", provider=provider, model="m"
        )
        assert decision["verdict"] == "done"
        assert not decision["should_continue"]
        assert mgr.state.status == "done"
        assert mgr.state.turns_used == 1

    @pytest.mark.asyncio
    async def test_continue_returns_prompt(self, tmp_path):
        mgr = GoalManager(_make_session(tmp_path))
        mgr.set("X")
        provider = _FakeProvider(
            content_to_return='{"done": false, "reason": "need next step"}'
        )
        decision = await mgr.evaluate_after_turn(
            "partial progress", provider=provider, model="m"
        )
        assert decision["verdict"] == "continue"
        assert decision["should_continue"]
        assert "X" in decision["continuation_prompt"]

    @pytest.mark.asyncio
    async def test_budget_exhaustion_pauses(self, tmp_path):
        mgr = GoalManager(_make_session(tmp_path))
        mgr.set("X", max_turns=2)
        provider = _FakeProvider(
            content_to_return='{"done": false, "reason": "grinding"}'
        )
        # Turn 1
        d1 = await mgr.evaluate_after_turn("r1", provider=provider, model="m")
        assert d1["should_continue"] is True
        # Turn 2 — hits budget
        d2 = await mgr.evaluate_after_turn("r2", provider=provider, model="m")
        assert d2["should_continue"] is False
        assert d2["status"] == "paused"
        assert mgr.state.status == "paused"
        assert "budget exhausted" in (mgr.state.paused_reason or "")

    @pytest.mark.asyncio
    async def test_judge_failure_keeps_going(self, tmp_path):
        """Fail-open semantics: broken judge doesn't wedge progress."""
        mgr = GoalManager(_make_session(tmp_path))
        mgr.set("X")
        provider = _FakeProvider(should_raise=RuntimeError("boom"))
        decision = await mgr.evaluate_after_turn(
            "anything", provider=provider, model="m"
        )
        assert decision["verdict"] == "continue"
        assert decision["should_continue"] is True
