"""Tar.gz snapshots of the workspace skills tree with transactional rollback.

Ported from Hermes' ``agent/curator_backup.py``, trimmed for nanobot:
  - no hub / bundled manifest (nanobot only distinguishes workspace vs builtin,
    and this module only touches workspace)
  - no cron-link restoration (cron jobs in nanobot don't reference skills by id)

Layout:
  <workspace>/skills/                      # the live tree
  <workspace>/skills/.curator_backups/     # snapshot root (excluded from snapshots)
  <workspace>/skills/.curator_backups/<utc-iso>/
      skills.tar.gz
      manifest.json                        # {reason, count, created_at, size}
  <workspace>/skills/.archive/             # archived skills (excluded from snapshots)

Rollback is transactional:
  1. Pre-rollback safety snapshot (so rollback itself is undoable)
  2. mv current top-level entries into a staging dir
  3. Extract the target snapshot into skills/
  4. On any failure, mv staged contents back
"""

from __future__ import annotations

import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

# Top-level entries under skills/ that must never be included in a snapshot
# or displaced during a rollback. These are curator-owned.
_EXCLUDE_TOP_LEVEL = {".curator_backups", ".archive"}

_BACKUPS_SUBDIR = ".curator_backups"
_ARCHIVE_NAME = "skills.tar.gz"
_MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _skills_dir(workspace: Path) -> Path:
    return workspace / "skills"


def _backups_dir(workspace: Path) -> Path:
    return _skills_dir(workspace) / _BACKUPS_SUBDIR


def _utc_id(now: datetime | None = None) -> str:
    """Return a safe-for-filesystem UTC timestamp id (``YYYYMMDDTHHMMSSZ``)."""
    if now is None:
        now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def _count_skill_files(base: Path) -> int:
    if not base.exists():
        return 0
    n = 0
    for entry in base.iterdir():
        if entry.name in _EXCLUDE_TOP_LEVEL:
            continue
        if not entry.is_dir():
            continue
        if (entry / "SKILL.md").exists():
            n += 1
    return n


def _write_manifest(
    dest: Path,
    reason: str,
    archive_path: Path,
    skill_count: int,
) -> None:
    try:
        archive_size = archive_path.stat().st_size if archive_path.exists() else 0
    except OSError:
        archive_size = 0
    manifest = {
        "id": dest.name,
        "reason": reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "skill_count": skill_count,
        "archive_size_bytes": archive_size,
    }
    try:
        (dest / _MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    except OSError as e:
        logger.debug("Failed to write manifest: {}", e)


def _read_manifest(snap_dir: Path) -> dict[str, Any]:
    manifest_file = snap_dir / _MANIFEST_NAME
    if not manifest_file.exists():
        return {"id": snap_dir.name, "reason": "(no manifest)"}
    try:
        return json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"id": snap_dir.name, "reason": "(manifest unreadable)"}


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def snapshot_skills(
    workspace: Path,
    *,
    reason: str = "manual",
    keep: int = 5,
) -> Path | None:
    """Create a ``tar.gz`` snapshot of ``<workspace>/skills/`` and prune old ones.

    Returns the snapshot directory, or ``None`` if skipped (skills dir missing
    or an IO error occurred — logged at debug; callers should treat ``None``
    as "no snapshot taken, continue without rolling back").
    """
    skills = _skills_dir(workspace)
    if not skills.exists():
        logger.debug("No skills dir — nothing to back up")
        return None

    backups = _backups_dir(workspace)
    try:
        backups.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.debug("Failed to create backups dir {}: {}", backups, e)
        return None

    # Uniquify: if a snapshot with the same second already exists, append
    # a counter. Avoids clobbering when two runs fire in the same second.
    base_id = _utc_id()
    snap_id = base_id
    counter = 1
    while (backups / snap_id).exists():
        snap_id = f"{base_id}-{counter:02d}"
        counter += 1

    dest = backups / snap_id
    try:
        dest.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        logger.debug("Failed to create snapshot dir {}: {}", dest, e)
        return None

    archive = dest / _ARCHIVE_NAME
    try:
        with tarfile.open(archive, "w:gz", compresslevel=6) as tf:
            for entry in sorted(skills.iterdir()):
                if entry.name in _EXCLUDE_TOP_LEVEL:
                    continue
                tf.add(str(entry), arcname=entry.name, recursive=True)
        _write_manifest(dest, reason, archive, _count_skill_files(skills))
    except (OSError, tarfile.TarError) as e:
        logger.debug("Snapshot failed: {}", e)
        shutil.rmtree(dest, ignore_errors=True)
        return None

    _prune_old(workspace, keep=keep)
    logger.info("Skill snapshot created: {} ({})", snap_id, reason)
    return dest


def _prune_old(workspace: Path, *, keep: int) -> list[str]:
    """Delete regular snapshots beyond the newest ``keep``.

    Pre-rollback safety snapshots (prefixed via their reason string) count
    against the same limit — that mirrors Hermes and keeps disk usage bounded
    even if the user does many rollbacks.
    """
    if keep < 0:
        keep = 0
    backups = _backups_dir(workspace)
    if not backups.exists():
        return []
    all_snaps = sorted(
        (p for p in backups.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    deleted: list[str] = []
    for snap in all_snaps[keep:]:
        try:
            shutil.rmtree(snap)
            deleted.append(snap.name)
        except OSError as e:
            logger.debug("Failed to prune snapshot {}: {}", snap.name, e)
    return deleted


# ---------------------------------------------------------------------------
# Listing / resolving
# ---------------------------------------------------------------------------

def list_backups(workspace: Path) -> list[dict[str, Any]]:
    """Return all snapshots, newest first."""
    backups = _backups_dir(workspace)
    if not backups.exists():
        return []
    out: list[dict[str, Any]] = []
    for snap in sorted(backups.iterdir(), reverse=True):
        if not snap.is_dir():
            continue
        if not (snap / _ARCHIVE_NAME).exists():
            continue
        out.append(_read_manifest(snap))
    return out


def _resolve_backup(workspace: Path, backup_id: str | None) -> Path | None:
    """Resolve a snapshot directory by id, or return the newest regular one."""
    backups = _backups_dir(workspace)
    if not backups.exists():
        return None
    if backup_id:
        candidate = backups / backup_id
        if candidate.is_dir() and (candidate / _ARCHIVE_NAME).exists():
            return candidate
        return None
    # Newest regular snapshot (exclude pre-rollback ones so the default
    # rollback target is always a user-sanctioned snapshot).
    regulars = [
        p for p in backups.iterdir()
        if p.is_dir() and (p / _ARCHIVE_NAME).exists()
        and not _read_manifest(p).get("reason", "").startswith("pre-rollback")
    ]
    if not regulars:
        return None
    return sorted(regulars, key=lambda p: p.name, reverse=True)[0]


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def rollback(
    workspace: Path,
    backup_id: str | None = None,
) -> tuple[bool, str, Path | None]:
    """Restore ``<workspace>/skills/`` from a snapshot.

    Returns ``(ok, message, snapshot_path)``.
    """
    target = _resolve_backup(workspace, backup_id)
    if target is None:
        hint = f" for id {backup_id!r}" if backup_id else ""
        return (
            False,
            f"no matching backup found{hint} (use list_backups to see options)",
            None,
        )
    archive = target / _ARCHIVE_NAME
    if not archive.exists():
        return (False, f"snapshot {target.name} has no {_ARCHIVE_NAME}", None)

    skills = _skills_dir(workspace)
    skills.mkdir(parents=True, exist_ok=True)
    backups = _backups_dir(workspace)
    backups.mkdir(parents=True, exist_ok=True)

    # Step 1: pre-rollback safety snapshot. Failure here aborts the rollback.
    try:
        snapshot_skills(
            workspace, reason=f"pre-rollback to {target.name}", keep=9999
        )
    except Exception as e:  # noqa: BLE001 — best-effort, any failure blocks us
        return (False, f"pre-rollback safety snapshot failed: {e}", None)

    # Step 2: stage current entries into an internal dir so we can extract
    # into an empty tree. mv (not rm) so we can restore on failure.
    staged = backups / f".rollback-staging-{_utc_id()}"
    try:
        staged.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        return (False, f"failed to create staging dir: {e}", None)

    moved: list[tuple[Path, Path]] = []
    try:
        for entry in list(skills.iterdir()):
            if entry.name in _EXCLUDE_TOP_LEVEL:
                continue
            dest = staged / entry.name
            shutil.move(str(entry), str(dest))
            moved.append((entry, dest))
    except OSError as e:
        for orig, dest in moved:
            try:
                shutil.move(str(dest), str(orig))
            except OSError:
                pass
        shutil.rmtree(staged, ignore_errors=True)
        return (False, f"failed to stage current skills: {e}", None)

    # Step 3: extract snapshot into skills/. Defend against path traversal
    # and absolute paths even on Python < 3.12 where ``filter="data"`` isn't
    # available.
    try:
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                name = member.name
                if name.startswith("/") or ".." in Path(name).parts:
                    raise tarfile.TarError(
                        f"refusing to extract unsafe path: {name!r}"
                    )
            try:
                tf.extractall(str(skills), filter="data")  # type: ignore[call-arg]
            except TypeError:  # pragma: no cover — Python < 3.12
                tf.extractall(str(skills))
    except (OSError, tarfile.TarError) as e:
        # Best-effort: move staged contents back so the user isn't left
        # with an empty skills dir.
        for orig, dest in moved:
            try:
                shutil.move(str(dest), str(orig))
            except OSError:
                pass
        shutil.rmtree(staged, ignore_errors=True)
        return (False, f"snapshot extract failed (state restored): {e}", None)

    # Step 4: success — drop staging. The user's undo handle is the
    # pre-rollback safety snapshot taken in step 1.
    shutil.rmtree(staged, ignore_errors=True)
    logger.info("Skill rollback: restored from {}", target.name)
    return (True, f"restored from snapshot {target.name}", target)


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f}TB"


def summarize_backups(workspace: Path) -> str:
    entries = list_backups(workspace)
    if not entries:
        return "No snapshots."
    lines: list[str] = []
    for m in entries:
        size = format_size(int(m.get("archive_size_bytes", 0)))
        count = m.get("skill_count", "?")
        reason = m.get("reason", "(no reason)")
        lines.append(f"- {m.get('id')}: {count} skill(s), {size} — {reason}")
    return "\n".join(lines)
