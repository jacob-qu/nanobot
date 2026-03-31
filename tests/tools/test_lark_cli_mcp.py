"""Tests for lark-cli MCP Server."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

import lark_cli_mcp
from lark_cli_mcp import (
    run_lark_cli,
    calendar_agenda,
    calendar_create_event,
    calendar_search,
    calendar_free_busy,
    task_list,
    task_create,
    task_complete,
    task_update,
    doc_search,
    doc_read,
    doc_create,
    raw,
)


# ---------------------------------------------------------------------------
# run_lark_cli tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_subprocess():
    """Fixture that patches asyncio.create_subprocess_exec."""

    def _make(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
        proc.returncode = returncode

        async def fake_exec(*args, **kwargs):
            return proc

        patcher = patch("lark_cli_mcp.asyncio.create_subprocess_exec", side_effect=fake_exec)
        # Patch shutil.which so the function doesn't early-return when lark-cli is absent
        which_patcher = patch.object(lark_cli_mcp.shutil, "which", return_value="/usr/bin/lark-cli")
        which_patcher.start()
        mock = patcher.start()
        return patcher, mock, proc, which_patcher

    return _make


@pytest.mark.asyncio
async def test_run_lark_cli_success(mock_subprocess):
    """Successful CLI call returns decoded stdout."""
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"items": []}', returncode=0)
    try:
        result = await run_lark_cli("calendar", "+agenda")
        assert result == '{"items": []}'
        # Verify --output json and --as user are appended
        call_args = mock.call_args[0]
        assert any("lark-cli" in a for a in call_args)
        assert "--output" in call_args
        assert "json" in call_args
        assert "--as" in call_args
        assert "user" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_run_lark_cli_error(mock_subprocess):
    """Non-zero exit code returns error with stderr."""
    patcher, mock, proc, which_patcher = mock_subprocess(stderr=b"auth required", returncode=1)
    try:
        result = await run_lark_cli("calendar", "+agenda")
        assert result.startswith("Error:")
        assert "auth required" in result
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_run_lark_cli_timeout():
    """Timeout returns timeout error message."""
    proc = AsyncMock()

    async def slow_communicate():
        await asyncio.sleep(10)
        return (b"", b"")

    proc.communicate = slow_communicate
    proc.returncode = 0
    proc.kill = lambda: None
    proc.wait = AsyncMock()

    async def fake_exec(*args, **kwargs):
        return proc

    with patch("lark_cli_mcp.asyncio.create_subprocess_exec", side_effect=fake_exec):
        with patch.object(lark_cli_mcp.shutil, "which", return_value="/usr/bin/lark-cli"):
            result = await run_lark_cli("calendar", "+agenda", timeout=0.01)
            assert "timed out" in result.lower() or "timeout" in result.lower()


# ---------------------------------------------------------------------------
# Calendar tools tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calendar_agenda_default(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(
        stdout=b'[{"summary":"Standup","start":"09:00","end":"09:30"}]'
    )
    try:
        result = await calendar_agenda()
        assert "Standup" in result
        call_args = mock.call_args[0]
        assert "calendar" in call_args
        assert "+agenda" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_calendar_agenda_multi_day(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'[]')
    try:
        result = await calendar_agenda(days=3)
        call_args = mock.call_args[0]
        assert "--start" in call_args
        assert "--end" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_calendar_create_event_basic(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"event_id":"ev_123"}')
    try:
        result = await calendar_create_event(
            summary="Team Sync",
            start_time="2026-04-02T14:00+08:00",
            end_time="2026-04-02T15:00+08:00",
        )
        assert "ev_123" in result
        call_args = mock.call_args[0]
        assert "+create" in call_args
        assert "--summary" in call_args
        assert "Team Sync" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_calendar_create_event_with_attendees(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"event_id":"ev_456"}')
    try:
        result = await calendar_create_event(
            summary="Review",
            start_time="2026-04-02T14:00+08:00",
            end_time="2026-04-02T15:00+08:00",
            attendees=["ou_aaa", "ou_bbb"],
        )
        call_args = mock.call_args[0]
        assert "--attendee-ids" in call_args
        assert "ou_aaa,ou_bbb" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_calendar_search(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'[]')
    try:
        result = await calendar_search(query="standup")
        call_args = mock.call_args[0]
        assert "calendar" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_calendar_free_busy(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"busy":[]}')
    try:
        result = await calendar_free_busy(
            user_ids=["ou_aaa"],
            start_time="2026-04-02T09:00+08:00",
            end_time="2026-04-02T18:00+08:00",
        )
        call_args = mock.call_args[0]
        assert "+freebusy" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


# ---------------------------------------------------------------------------
# Task tools tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_list_default(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(
        stdout=b'[{"summary":"Buy milk","completed":false}]'
    )
    try:
        result = await task_list()
        assert "Buy milk" in result
        call_args = mock.call_args[0]
        assert "+get-my-tasks" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_task_list_with_count(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'[]')
    try:
        result = await task_list(count=5)
        call_args = mock.call_args[0]
        assert "--page-limit" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_task_create_minimal(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"task_id":"t_123"}')
    try:
        result = await task_create(summary="Write tests")
        assert "t_123" in result
        call_args = mock.call_args[0]
        assert "+create" in call_args
        assert "--summary" in call_args
        assert "Write tests" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_task_create_full(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"task_id":"t_456"}')
    try:
        result = await task_create(
            summary="Deploy v2",
            due_time="2026-04-05",
            description="Deploy the new release",
        )
        call_args = mock.call_args[0]
        assert "--due" in call_args
        assert "--description" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_task_complete(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"status":"completed"}')
    try:
        result = await task_complete(task_id="t_123")
        call_args = mock.call_args[0]
        assert "+complete" in call_args
        assert "--task-id" in call_args
        assert "t_123" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_task_update(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"task_id":"t_123"}')
    try:
        result = await task_update(task_id="t_123", summary="Updated title")
        call_args = mock.call_args[0]
        assert "+update" in call_args
        assert "--task-id" in call_args
        assert "--summary" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


# ---------------------------------------------------------------------------
# Doc tools tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_search(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(
        stdout=b'[{"title":"Project Plan","url":"https://..."}]'
    )
    try:
        result = await doc_search(query="project plan")
        assert "Project Plan" in result
        call_args = mock.call_args[0]
        assert "docs" in call_args
        assert "+search" in call_args
        assert "--query" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_doc_search_with_count(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'[]')
    try:
        result = await doc_search(query="notes", count=5)
        call_args = mock.call_args[0]
        assert "--page-size" in call_args
        assert "5" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_doc_read(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"content":"# Hello"}')
    try:
        result = await doc_read(document_id="Z1Fjxxx")
        assert "Hello" in result
        call_args = mock.call_args[0]
        assert "+fetch" in call_args
        assert "--doc" in call_args
        assert "Z1Fjxxx" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_doc_create_minimal(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"document_id":"doc_123"}')
    try:
        result = await doc_create(title="New Doc")
        assert "doc_123" in result
        call_args = mock.call_args[0]
        assert "+create" in call_args
        assert "--title" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_doc_create_with_content_and_folder(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"document_id":"doc_456"}')
    try:
        result = await doc_create(
            title="Meeting Notes",
            content="## Discussion\n\n- Item 1",
            folder_id="fldcnXXX",
        )
        call_args = mock.call_args[0]
        assert "--markdown" in call_args
        assert "--folder-token" in call_args
        assert "fldcnXXX" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


# ---------------------------------------------------------------------------
# Raw tool tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_simple_command(mock_subprocess):
    patcher, mock, proc, which_patcher = mock_subprocess(stdout=b'{"ok":true}')
    try:
        result = await raw(command="im +messages-send --chat-id oc_xxx --text hello")
        assert "ok" in result
        call_args = mock.call_args[0]
        assert "im" in call_args
    finally:
        patcher.stop()
        which_patcher.stop()


@pytest.mark.asyncio
async def test_raw_empty_command():
    result = await raw(command="")
    assert "Error" in result


@pytest.mark.asyncio
async def test_lark_cli_not_installed(monkeypatch):
    import lark_cli_mcp as mod
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)

    result = await mod.run_lark_cli("calendar", "+agenda")
    assert "not installed" in result.lower()
    assert "npm install" in result
