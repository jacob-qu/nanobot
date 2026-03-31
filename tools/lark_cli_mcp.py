"""MCP Server wrapping lark-cli for nanobot Agent integration.

Provides structured tools for calendar, tasks, and docs operations,
plus a raw fallback tool for arbitrary lark-cli commands.

Usage:
    python3 tools/lark_cli_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
from asyncio.subprocess import PIPE
from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("lark-cli")

# ---------------------------------------------------------------------------
# Core CLI helper
# ---------------------------------------------------------------------------


async def run_lark_cli(*args: str, timeout: int = 25) -> str:
    """Execute a lark-cli command and return its stdout.

    All calls are forced to ``--output json --as user``.
    Returns an error string (starting with ``Error:``) on failure.
    """
    lark_cli = shutil.which("lark-cli")
    if not lark_cli:
        return "Error: lark-cli is not installed. Run: npm install -g @larksuite/cli"

    cmd = [lark_cli, *args, "--output", "json", "--as", "user"]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"Error: lark-cli timed out after {timeout}s"

    if proc.returncode != 0:
        err = stderr.decode().strip() if stderr else "unknown error"
        if "login" in err.lower() or "auth" in err.lower() or "token" in err.lower():
            return (
                f"Error: {err}\n\nPlease authenticate first: "
                "lark-cli auth login --domain calendar,task,doc,drive"
            )
        return f"Error: {err}"

    return stdout.decode().strip()


# ---------------------------------------------------------------------------
# Calendar tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def calendar_agenda(days: int = 1) -> str:
    """View calendar agenda for the next N days. Returns a list of upcoming events."""
    args = ["calendar", "+agenda"]
    if days > 1:
        today = date.today()
        args += ["--start", today.isoformat(), "--end", (today + timedelta(days=days)).isoformat()]
    return await run_lark_cli(*args)


@mcp.tool()
async def calendar_create_event(
    summary: str,
    start_time: str,
    end_time: str,
    description: str = "",
    attendees: list[str] | None = None,
) -> str:
    """Create a calendar event. Times must be ISO 8601 format (e.g. 2026-04-02T14:00+08:00)."""
    args = ["calendar", "+create", "--summary", summary, "--start", start_time, "--end", end_time]
    if description:
        args += ["--description", description]
    if attendees:
        args += ["--attendee-ids", ",".join(attendees)]
    return await run_lark_cli(*args)


@mcp.tool()
async def calendar_search(
    query: str,
    start_time: str = "",
    end_time: str = "",
) -> str:
    """Search calendar events by keyword. Optionally filter by time range."""
    params: dict = {"query": query}
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time
    return await run_lark_cli(
        "calendar", "events", "search",
        "--params", json.dumps(params, ensure_ascii=False),
    )


@mcp.tool()
async def calendar_free_busy(
    user_ids: list[str],
    start_time: str,
    end_time: str,
) -> str:
    """Check free/busy status for users in a time range."""
    args = ["calendar", "+freebusy", "--start", start_time, "--end", end_time]
    if user_ids:
        args += ["--user-id", ",".join(user_ids)]
    return await run_lark_cli(*args)


# ---------------------------------------------------------------------------
# Task tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def task_list(count: int = 20) -> str:
    """List my pending tasks. Returns task summaries with IDs."""
    args = ["task", "+get-my-tasks"]
    if count != 20:
        args += ["--page-limit", str(count)]
    return await run_lark_cli(*args)


@mcp.tool()
async def task_create(
    summary: str,
    due_time: str = "",
    description: str = "",
) -> str:
    """Create a new task. due_time supports ISO 8601, YYYY-MM-DD, or relative like +2d."""
    args = ["task", "+create", "--summary", summary]
    if due_time:
        args += ["--due", due_time]
    if description:
        args += ["--description", description]
    return await run_lark_cli(*args)


@mcp.tool()
async def task_complete(task_id: str) -> str:
    """Mark a task as completed."""
    return await run_lark_cli("task", "+complete", "--task-id", task_id)


@mcp.tool()
async def task_update(
    task_id: str,
    summary: str = "",
    due_time: str = "",
    description: str = "",
) -> str:
    """Update an existing task's fields."""
    args = ["task", "+update", "--task-id", task_id]
    if summary:
        args += ["--summary", summary]
    if due_time:
        args += ["--due", due_time]
    if description:
        args += ["--description", description]
    return await run_lark_cli(*args)


# ---------------------------------------------------------------------------
# Doc tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def doc_search(query: str, count: int = 10) -> str:
    """Search documents and drive files by keyword."""
    args = ["docs", "+search", "--query", query]
    if count != 10:
        args += ["--page-size", str(count)]
    return await run_lark_cli(*args)


@mcp.tool()
async def doc_read(document_id: str) -> str:
    """Read a document's content. Accepts document token or full URL."""
    return await run_lark_cli("docs", "+fetch", "--doc", document_id)


@mcp.tool()
async def doc_create(
    title: str,
    content: str = "",
    folder_id: str = "",
) -> str:
    """Create a new document. Content is markdown format."""
    args = ["docs", "+create", "--title", title]
    args += ["--markdown", content]
    if folder_id:
        args += ["--folder-token", folder_id]
    return await run_lark_cli(*args)


# ---------------------------------------------------------------------------
# Raw fallback tool
# ---------------------------------------------------------------------------


@mcp.tool()
async def raw(command: str) -> str:
    """Execute any lark-cli command directly. Pass the subcommand and flags
    without the 'lark-cli' prefix. Example: 'im +messages-send --chat-id oc_xxx --text hello'
    """
    if not command.strip():
        return "Error: command cannot be empty"
    parts = command.strip().split()
    return await run_lark_cli(*parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
