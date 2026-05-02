"""Background maintenance for agent-created skills.

Runs in two phases:

1. **Reconcile** — for every workspace skill not yet tracked in ``skill_usage``,
   create a row with ``created_at = mtime(SKILL.md)``. Hand-written skills
   that predate ``manage_skill``打点 still participate in state transitions.

2. **Automatic transitions** (pure, no LLM) —
   * Unused for ``stale_after_days`` (default 30) → ``STATE_STALE``
   * Unused for ``archive_after_days`` (default 90) → archived to
     ``skills/.archive/<name>/``, state set to ``STATE_ARCHIVED``
   * Previously STALE skill used again → reactivated to ``STATE_ACTIVE``
   * Pinned skills (frontmatter ``pinned: true``) are skipped entirely

Gating mirrors Hermes:
  * ``interval_hours`` (default 168 / 7 days) since last real run
  * ``min_idle_hours`` (default 2) of agent idle time — enforced at the
    call site, because only the caller knows how to measure "idle"
  * First observation of ``last_run_at = None`` seeds state and defers the
    first real pass by one full interval (opt-out window)

The LLM review pass (Hermes' "umbrella building") is NOT implemented here —
it requires a forked agent with a dedicated auxiliary model. Can be added
later; the state-machine half is independently useful.
"""

from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from nanobot.agent import curator_backup
from nanobot.agent.skill_usage import (
    STATE_ACTIVE,
    STATE_ARCHIVED,
    STATE_STALE,
    SkillUsageStore,
)

# ---------------------------------------------------------------------------
# State file (curator-run-level metadata: last_run_at, paused, ...)
# ---------------------------------------------------------------------------

_STATE_FILENAME = "curator_state.json"
_STATE_LOCK = threading.Lock()


def _state_path(workspace: Path) -> Path:
    return workspace / "memory" / _STATE_FILENAME


def _default_state() -> dict[str, Any]:
    return {
        "last_run_at": None,
        "run_count": 0,
        "last_run_summary": "",
        "last_run_duration_seconds": 0.0,
        "paused": False,
    }


def load_state(workspace: Path) -> dict[str, Any]:
    path = _state_path(workspace)
    if not path.exists():
        return _default_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("curator state read failed: {}", e)
        return _default_state()
    merged = _default_state()
    if isinstance(data, dict):
        merged.update(data)
    return merged


def save_state(workspace: Path, data: dict[str, Any]) -> None:
    with _STATE_LOCK:
        path = _state_path(workspace)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as e:
            logger.debug("curator state write failed: {}", e)


def set_paused(workspace: Path, paused: bool) -> None:
    state = load_state(workspace)
    state["paused"] = bool(paused)
    save_state(workspace, state)


def is_paused(workspace: Path) -> bool:
    return bool(load_state(workspace).get("paused", False))


# ---------------------------------------------------------------------------
# Config accessors (the caller passes in CuratorConfig; we just read it)
# ---------------------------------------------------------------------------

def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Skill enumeration
# ---------------------------------------------------------------------------

_WORKSPACE_EXCLUDE = {".curator_backups", ".archive"}


def _workspace_skill_dirs(workspace: Path) -> list[Path]:
    """All skill directories under ``<workspace>/skills/`` (excluding archive)."""
    skills = workspace / "skills"
    if not skills.exists():
        return []
    out = []
    for entry in skills.iterdir():
        if entry.name in _WORKSPACE_EXCLUDE:
            continue
        if not entry.is_dir():
            continue
        if not (entry / "SKILL.md").exists():
            continue
        out.append(entry)
    return out


def _read_frontmatter(skill_dir: Path) -> dict[str, Any]:
    skill_md = skill_dir / "SKILL.md"
    try:
        content = skill_md.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not content.startswith("---"):
        return {}
    try:
        body = content.split("---", 2)
        if len(body) < 3:
            return {}
        parsed = yaml.safe_load(body[1])
    except yaml.YAMLError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _is_pinned(skill_dir: Path) -> bool:
    """Read frontmatter and look for ``pinned: true`` at top level or under
    ``nanobot.pinned`` / ``openclaw.pinned``."""
    fm = _read_frontmatter(skill_dir)
    if not fm:
        return False
    if bool(fm.get("pinned")):
        return True
    for key in ("nanobot", "openclaw"):
        sub = fm.get(key)
        if isinstance(sub, dict) and bool(sub.get("pinned")):
            return True
    return False


def _mtime_iso(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except OSError:
        ts = datetime.now(timezone.utc).timestamp()
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def reconcile_usage(workspace: Path, usage: SkillUsageStore) -> int:
    """Populate ``skill_usage`` rows for workspace skills that don't have one.

    Returns the number of new rows created. Does not touch existing rows.
    Uses ``mtime(SKILL.md)`` as the synthetic created_at for hand-written
    skills that predate打点 — better than "now" because it reflects real
    age for archival decisions.
    """
    created = 0
    for skill_dir in _workspace_skill_dirs(workspace):
        name = skill_dir.name
        if usage.get_record(name) is not None:
            continue
        ts = _mtime_iso(skill_dir / "SKILL.md")
        # Seed the row directly so last_activity_at reflects mtime, not "now".
        with usage._lock:  # noqa: SLF001 — intentional, keeps module-private
            db = usage._get_db()  # noqa: SLF001
            db.execute(
                "INSERT OR IGNORE INTO skill_usage "
                "(name, state, created_at, last_activity_at) "
                "VALUES (?, ?, ?, ?)",
                (name, STATE_ACTIVE, ts, ts),
            )
            db.commit()
        created += 1
    return created


# ---------------------------------------------------------------------------
# Archive (filesystem move)
# ---------------------------------------------------------------------------

def archive_skill_dir(workspace: Path, name: str) -> tuple[bool, str]:
    """Move ``skills/<name>`` into ``skills/.archive/<name>[_<ts>]``.

    Returns ``(ok, message)``. Never raises; a failure leaves the directory
    in place and the caller can decide what to do.
    """
    src = workspace / "skills" / name
    if not src.exists():
        return (False, f"skill {name!r} does not exist")
    if not src.is_dir():
        return (False, f"skill {name!r} is not a directory")

    archive_root = workspace / "skills" / ".archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    dst = archive_root / name
    if dst.exists():
        # Name collision with previously-archived skill of same name —
        # append a timestamp so both are preserved.
        ts = _now().strftime("%Y%m%dT%H%M%SZ")
        dst = archive_root / f"{name}_{ts}"

    try:
        shutil.move(str(src), str(dst))
    except OSError as e:
        return (False, f"archive move failed: {e}")
    return (True, f"archived to {dst.relative_to(workspace)}")


# ---------------------------------------------------------------------------
# Automatic transitions (pure; no LLM)
# ---------------------------------------------------------------------------

def apply_automatic_transitions(
    workspace: Path,
    usage: SkillUsageStore,
    *,
    stale_after_days: int,
    archive_after_days: int,
    now: datetime | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Walk every workspace skill and apply state transitions.

    Returns a dict with counters + per-skill action list. In ``dry_run``
    mode no filesystem or DB mutations happen — the returned actions show
    what *would* happen.
    """
    if now is None:
        now = _now()
    from datetime import timedelta
    stale_cutoff = now - timedelta(days=stale_after_days)
    archive_cutoff = now - timedelta(days=archive_after_days)

    counts = {
        "checked": 0,
        "skipped_pinned": 0,
        "marked_stale": 0,
        "archived": 0,
        "reactivated": 0,
        "unchanged": 0,
    }
    actions: list[dict[str, Any]] = []

    for skill_dir in _workspace_skill_dirs(workspace):
        name = skill_dir.name
        counts["checked"] += 1

        if _is_pinned(skill_dir):
            counts["skipped_pinned"] += 1
            actions.append({"name": name, "action": "skip", "reason": "pinned"})
            continue

        row = usage.get_record(name)
        if row is None:
            # Shouldn't happen after reconcile_usage, but be defensive.
            counts["unchanged"] += 1
            continue

        anchor = (
            _parse_iso(row.get("last_activity_at"))
            or _parse_iso(row.get("created_at"))
            or now
        )
        current = row.get("state", STATE_ACTIVE)

        if anchor <= archive_cutoff and current != STATE_ARCHIVED:
            if not dry_run:
                ok, msg = archive_skill_dir(workspace, name)
                if ok:
                    usage.set_state(name, STATE_ARCHIVED)
                else:
                    actions.append({"name": name, "action": "archive_failed", "reason": msg})
                    continue
            counts["archived"] += 1
            actions.append({
                "name": name, "action": "archive",
                "anchor_age_days": (now - anchor).days,
            })
        elif anchor <= stale_cutoff and current == STATE_ACTIVE:
            if not dry_run:
                usage.set_state(name, STATE_STALE)
            counts["marked_stale"] += 1
            actions.append({
                "name": name, "action": "mark_stale",
                "anchor_age_days": (now - anchor).days,
            })
        elif anchor > stale_cutoff and current == STATE_STALE:
            if not dry_run:
                usage.set_state(name, STATE_ACTIVE)
            counts["reactivated"] += 1
            actions.append({"name": name, "action": "reactivate"})
        else:
            counts["unchanged"] += 1

    return {"counts": counts, "actions": actions}


# ---------------------------------------------------------------------------
# Gating + top-level run
# ---------------------------------------------------------------------------

def should_run_now(
    workspace: Path,
    *,
    enabled: bool,
    interval_hours: int,
    now: datetime | None = None,
) -> bool:
    """Return True if the interval gate passes. Idle gate is enforced by caller.

    First-observation behavior: if ``last_run_at`` is missing we seed it to
    now (defers the first real pass by one full interval). This gives the
    user an opt-out window before the curator ever mutates anything.
    """
    if not enabled:
        return False
    if is_paused(workspace):
        return False
    state = load_state(workspace)
    last = _parse_iso(state.get("last_run_at"))
    if last is None:
        if now is None:
            now = _now()
        state["last_run_at"] = now.isoformat()
        state["last_run_summary"] = (
            "deferred first run — curator seeded, will run after one interval"
        )
        save_state(workspace, state)
        return False
    if now is None:
        now = _now()
    from datetime import timedelta
    return (now - last) >= timedelta(hours=interval_hours)


def run_curator(
    workspace: Path,
    usage: SkillUsageStore,
    *,
    stale_after_days: int = 30,
    archive_after_days: int = 90,
    backup_keep: int = 5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute one curator pass (state machine only, no LLM).

    Steps:
      1. reconcile_usage — make sure every workspace skill has a row
      2. snapshot (skipped in dry_run)
      3. apply_automatic_transitions
      4. persist state (skipped in dry_run)

    Returns a structured report.
    """
    start = _now()
    result: dict[str, Any] = {
        "started_at": start.isoformat(),
        "dry_run": dry_run,
        "reconciled": 0,
        "snapshot": None,
        "transitions": None,
        "summary": "",
    }

    # 1. Reconcile any untracked workspace skills
    result["reconciled"] = reconcile_usage(workspace, usage)

    # 2. Pre-run snapshot (best-effort; never aborts the run)
    if not dry_run:
        snap = curator_backup.snapshot_skills(
            workspace, reason="pre-curator-run", keep=backup_keep,
        )
        if snap is not None:
            result["snapshot"] = snap.name

    # 3. State transitions
    tr = apply_automatic_transitions(
        workspace, usage,
        stale_after_days=stale_after_days,
        archive_after_days=archive_after_days,
        now=start,
        dry_run=dry_run,
    )
    result["transitions"] = tr

    # 4. Persist curator state (dry-run does not bump last_run_at)
    counts = tr["counts"]
    parts = []
    if counts["marked_stale"]: parts.append(f"{counts['marked_stale']} stale")
    if counts["archived"]:     parts.append(f"{counts['archived']} archived")
    if counts["reactivated"]:  parts.append(f"{counts['reactivated']} reactivated")
    if counts["skipped_pinned"]: parts.append(f"{counts['skipped_pinned']} pinned")
    summary = ", ".join(parts) if parts else "no changes"
    result["summary"] = summary

    state = load_state(workspace)
    if not dry_run:
        state["last_run_at"] = start.isoformat()
        state["run_count"] = int(state.get("run_count", 0)) + 1
    prefix = "dry-run: " if dry_run else ""
    state["last_run_summary"] = f"{prefix}{summary}"
    state["last_run_duration_seconds"] = (_now() - start).total_seconds()
    save_state(workspace, state)

    return result


def maybe_run_curator(
    workspace: Path,
    usage: SkillUsageStore,
    *,
    enabled: bool,
    interval_hours: int,
    min_idle_hours: float,
    stale_after_days: int = 30,
    archive_after_days: int = 90,
    backup_keep: int = 5,
    idle_for_seconds: float | None = None,
) -> dict[str, Any] | None:
    """Run a curator pass if enabled + interval + idle gates all pass.

    Returns the run result, or ``None`` if the run was skipped. Never raises
    — any internal error is logged and returns ``None`` so a failing curator
    can't take down the caller (cron tick, CLI session start, ...).
    """
    try:
        if not should_run_now(
            workspace, enabled=enabled, interval_hours=interval_hours,
        ):
            return None
        if idle_for_seconds is not None:
            min_idle_s = min_idle_hours * 3600.0
            if idle_for_seconds < min_idle_s:
                return None
        return run_curator(
            workspace, usage,
            stale_after_days=stale_after_days,
            archive_after_days=archive_after_days,
            backup_keep=backup_keep,
        )
    except Exception as e:  # noqa: BLE001 — top-level safety net
        logger.debug("maybe_run_curator failed: {}", e)
        return None
