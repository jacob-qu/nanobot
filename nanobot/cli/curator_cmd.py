"""CLI subcommands for the skill curator.

Exposes ``nanobot curator <subcommand>``:
  * ``status``        — one-line summary of last run + pinned/LRU
  * ``run``           — trigger a pass now (add ``--dry-run`` for preview)
  * ``pause`` / ``resume``
  * ``rollback``      — restore from a snapshot (newest regular by default)
  * ``backups``       — list available snapshots

All commands require a configured workspace (``nanobot onboard`` already run).
"""

from __future__ import annotations

import typer
from rich.console import Console

from nanobot.agent import curator, curator_backup
from nanobot.agent.skill_usage import SkillUsageStore
from nanobot.config.loader import load_config

app = typer.Typer(
    name="curator",
    help="Background maintenance for agent-created skills",
    no_args_is_help=True,
)
console = Console()


def _usage_store() -> tuple[SkillUsageStore, "Path"]:  # noqa: F821
    """Return ``(store, workspace_path)`` honoring the active config."""
    cfg = load_config()
    workspace = cfg.workspace_path
    return SkillUsageStore(workspace), workspace


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "(never)"
    # Trim microseconds for readability
    return ts.split(".", 1)[0].replace("T", " ")


@app.command()
def status() -> None:
    """Show last run, counts, pinned skills, and LRU top 5."""
    usage, workspace = _usage_store()
    st = curator.load_state(workspace)
    cfg = load_config().agents.defaults.curator

    console.print(f"[bold]Curator[/bold]  workspace={workspace}")
    console.print(
        f"  enabled = {cfg.enabled}  "
        f"paused = {st.get('paused', False)}  "
        f"interval = {cfg.interval_hours}h  "
        f"idle ≥ {cfg.min_idle_hours}h"
    )
    console.print(
        f"  thresholds: stale > {cfg.stale_after_days}d, "
        f"archive > {cfg.archive_after_days}d"
    )
    console.print(f"  last run : {_fmt_ts(st.get('last_run_at'))}")
    console.print(f"  run count: {st.get('run_count', 0)}")
    summary = st.get("last_run_summary") or "(none)"
    console.print(f"  summary  : {summary}")

    records = usage.all_records()
    by_state: dict[str, int] = {}
    for r in records:
        by_state[r["state"]] = by_state.get(r["state"], 0) + 1
    if records:
        console.print(
            f"\n  skills tracked: {len(records)}  "
            + "  ".join(f"{k}={v}" for k, v in sorted(by_state.items()))
        )

        lru = usage.least_recently_used(limit=5)
        if lru:
            console.print("\n  [dim]Least recently used:[/dim]")
            for row in lru:
                ts = _fmt_ts(row.get("last_activity_at"))
                console.print(
                    f"    - {row['name']} ({row['state']}, "
                    f"use={row.get('use_count', 0)}, last={ts})"
                )
    else:
        console.print("\n  (no skills tracked yet)")


@app.command()
def run(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview actions without mutating anything."
    ),
) -> None:
    """Execute one curator pass now, bypassing the interval/idle gates."""
    usage, workspace = _usage_store()
    cfg = load_config().agents.defaults.curator

    result = curator.run_curator(
        workspace, usage,
        stale_after_days=cfg.stale_after_days,
        archive_after_days=cfg.archive_after_days,
        backup_keep=cfg.backup_keep,
        dry_run=dry_run,
    )

    prefix = "[DRY-RUN] " if dry_run else ""
    console.print(f"{prefix}[bold]{result['summary']}[/bold]")
    if result.get("snapshot"):
        console.print(f"  pre-run snapshot: {result['snapshot']}")
    if result.get("reconciled"):
        console.print(f"  reconciled {result['reconciled']} untracked skill(s)")

    actions = (result.get("transitions") or {}).get("actions") or []
    if not actions:
        return
    console.print("\n  [dim]Actions:[/dim]")
    for a in actions:
        extra = ""
        if "anchor_age_days" in a:
            extra = f" (idle {a['anchor_age_days']}d)"
        console.print(f"    - {a['name']}: {a['action']}{extra}")


@app.command()
def pause() -> None:
    """Pause the curator — no runs until ``resume``."""
    _, workspace = _usage_store()
    curator.set_paused(workspace, True)
    console.print("[yellow]Curator paused.[/yellow]")


@app.command()
def resume() -> None:
    """Resume the curator."""
    _, workspace = _usage_store()
    curator.set_paused(workspace, False)
    console.print("[green]Curator resumed.[/green]")


@app.command()
def backups(
    limit: int = typer.Option(10, "--limit", "-n", help="Max entries to show"),
) -> None:
    """List available skill-tree snapshots (newest first)."""
    _, workspace = _usage_store()
    entries = curator_backup.list_backups(workspace)
    if not entries:
        console.print("No snapshots.")
        return
    for m in entries[:limit]:
        size = curator_backup.format_size(int(m.get("archive_size_bytes", 0)))
        console.print(
            f"  {m.get('id')}  "
            f"{m.get('skill_count', '?')} skill(s)  "
            f"{size}  "
            f"— {m.get('reason', '(no reason)')}"
        )


@app.command()
def rollback(
    backup_id: str | None = typer.Option(
        None, "--id", help="Specific snapshot id (default: newest regular)"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation prompt"
    ),
) -> None:
    """Restore ``skills/`` from a snapshot.

    A pre-rollback safety snapshot is always taken first, so this action
    itself is undoable by rolling forward to that one.
    """
    _, workspace = _usage_store()
    target_hint = backup_id or "(newest regular snapshot)"
    if not yes and not typer.confirm(
        f"Restore skills/ from {target_hint}?  This will replace the current tree."
    ):
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(1)

    ok, msg, path = curator_backup.rollback(workspace, backup_id=backup_id)
    if ok:
        console.print(f"[green]✓[/green] {msg}")
    else:
        console.print(f"[red]✗[/red] {msg}")
        raise typer.Exit(1)
