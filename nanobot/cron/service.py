"""Cron service for scheduling agent tasks."""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from loguru import logger

from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule, CronStore

# Maximum sleep between tick cycles (seconds).
# Keeps the service responsive to clock adjustments and system resume.
_MAX_TICK_INTERVAL_S = 60


def _now_ms() -> int:
    return int(time.time() * 1000)


def _is_6field(expr: str) -> bool:
    """Return True if expr has 6 space-separated fields (Quartz-style with seconds)."""
    return len(expr.strip().split()) == 6


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """Compute next run time in ms."""
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        # Next interval from now
        return now_ms + schedule.every_ms

    if schedule.kind == "cron" and schedule.expr:
        try:
            from zoneinfo import ZoneInfo

            from croniter import croniter
            # Use caller-provided reference time for deterministic scheduling
            base_time = now_ms / 1000
            tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
            base_dt = datetime.fromtimestamp(base_time, tz=tz)
            # Support Quartz-style 6-field expressions (seconds as first field)
            kwargs = {}
            if _is_6field(schedule.expr):
                kwargs["second_at_beginning"] = True
            cron = croniter(schedule.expr, base_dt, **kwargs)
            next_dt = cron.get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except Exception as e:
            logger.warning("Cron: failed to parse expression '{}': {}", schedule.expr, e)
            return None

    return None


def _validate_schedule_for_add(schedule: CronSchedule) -> None:
    """Validate schedule fields that would otherwise create non-runnable jobs."""
    if schedule.tz and schedule.kind != "cron":
        raise ValueError("tz can only be used with cron schedules")

    if schedule.kind == "cron" and schedule.tz:
        try:
            from zoneinfo import ZoneInfo

            ZoneInfo(schedule.tz)
        except Exception:
            raise ValueError(f"unknown timezone '{schedule.tz}'") from None

    if schedule.kind == "cron" and schedule.expr:
        try:
            from croniter import croniter

            kwargs = {}
            if _is_6field(schedule.expr):
                kwargs["second_at_beginning"] = True
            croniter(schedule.expr, **kwargs)
        except Exception as e:
            raise ValueError(f"invalid cron expression '{schedule.expr}': {e}") from None


def _append_audit_log(store_path: Path, job: CronJob, status: str, error: str | None = None) -> None:
    """Append a line to the cron audit log (sibling of jobs.json)."""
    try:
        log_path = store_path.parent / "audit.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
        parts = [ts, job.id, job.name, status]
        if error:
            parts.append(error[:200])
        line = " | ".join(parts) + "\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        logger.debug("Cron: audit log write failed: {}", e)


class CronService:
    """Service for managing and executing scheduled jobs."""

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None
    ):
        self.store_path = store_path
        self.on_job = on_job
        self._store: CronStore | None = None
        self._last_mtime: float = 0.0
        self._tick_task: asyncio.Task | None = None
        self._running = False

    def _load_store(self) -> CronStore:
        """Load jobs from disk. Reloads automatically if file was modified externally."""
        if self._store and self.store_path.exists():
            mtime = self.store_path.stat().st_mtime
            if mtime != self._last_mtime:
                logger.info("Cron: jobs.json modified externally, reloading")
                self._store = None
        if self._store:
            return self._store

        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                jobs = []
                for j in data.get("jobs", []):
                    jobs.append(CronJob(
                        id=j["id"],
                        name=j["name"],
                        enabled=j.get("enabled", True),
                        schedule=CronSchedule(
                            kind=j["schedule"]["kind"],
                            at_ms=j["schedule"].get("atMs"),
                            every_ms=j["schedule"].get("everyMs"),
                            expr=j["schedule"].get("expr"),
                            tz=j["schedule"].get("tz"),
                        ),
                        payload=CronPayload(
                            kind=j["payload"].get("kind", "agent_turn"),
                            message=j["payload"].get("message", ""),
                            deliver=j["payload"].get("deliver", False),
                            channel=j["payload"].get("channel"),
                            to=j["payload"].get("to"),
                        ),
                        state=CronJobState(
                            next_run_at_ms=j.get("state", {}).get("nextRunAtMs"),
                            last_run_at_ms=j.get("state", {}).get("lastRunAtMs"),
                            last_status=j.get("state", {}).get("lastStatus"),
                            last_error=j.get("state", {}).get("lastError"),
                        ),
                        created_at_ms=j.get("createdAtMs", 0),
                        updated_at_ms=j.get("updatedAtMs", 0),
                        delete_after_run=j.get("deleteAfterRun", False),
                    ))
                self._store = CronStore(jobs=jobs)
            except Exception as e:
                logger.warning("Failed to load cron store: {}", e)
                self._store = CronStore()
        else:
            self._store = CronStore()

        return self._store

    def _save_store(self) -> None:
        """Save jobs to disk."""
        if not self._store:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self._store.version,
            "jobs": [
                {
                    "id": j.id,
                    "name": j.name,
                    "enabled": j.enabled,
                    "schedule": {
                        "kind": j.schedule.kind,
                        "atMs": j.schedule.at_ms,
                        "everyMs": j.schedule.every_ms,
                        "expr": j.schedule.expr,
                        "tz": j.schedule.tz,
                    },
                    "payload": {
                        "kind": j.payload.kind,
                        "message": j.payload.message,
                        "deliver": j.payload.deliver,
                        "channel": j.payload.channel,
                        "to": j.payload.to,
                    },
                    "state": {
                        "nextRunAtMs": j.state.next_run_at_ms,
                        "lastRunAtMs": j.state.last_run_at_ms,
                        "lastStatus": j.state.last_status,
                        "lastError": j.state.last_error,
                    },
                    "createdAtMs": j.created_at_ms,
                    "updatedAtMs": j.updated_at_ms,
                    "deleteAfterRun": j.delete_after_run,
                }
                for j in self._store.jobs
            ]
        }

        self.store_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._last_mtime = self.store_path.stat().st_mtime
    
    async def start(self) -> None:
        """Start the cron service."""
        self._running = True
        self._load_store()
        await self._catch_up_missed_jobs()
        self._recompute_next_runs()
        self._save_store()
        self._start_tick_loop()
        logger.info("Cron service started with {} jobs", len(self._store.jobs if self._store else []))

    def stop(self) -> None:
        """Stop the cron service."""
        self._running = False
        if self._tick_task:
            self._tick_task.cancel()
            self._tick_task = None

    def _recompute_next_runs(self) -> None:
        """Recompute next run times for enabled jobs that need it.

        - 'every' jobs always get recomputed from now (relative scheduling).
        - 'cron'/'at' jobs only get computed if they don't already have a next_run_at_ms
          (to preserve persisted values for catch-up detection).
        """
        if not self._store:
            return
        now = _now_ms()
        for job in self._store.jobs:
            if not job.enabled:
                continue
            if job.schedule.kind == "every":
                job.state.next_run_at_ms = _compute_next_run(job.schedule, now)
            elif not job.state.next_run_at_ms:
                job.state.next_run_at_ms = _compute_next_run(job.schedule, now)

    async def _catch_up_missed_jobs(self) -> None:
        """Detect jobs whose next_run_at_ms is in the past (missed while offline) and execute them.

        Only applies to 'cron' and 'at' schedule types. Interval ('every') jobs simply
        get their next_run_at_ms recomputed from now, since replaying missed intervals
        is not meaningful.
        """
        if not self._store:
            return
        now = _now_ms()
        missed = []
        for j in self._store.jobs:
            if not j.enabled or not j.state.next_run_at_ms:
                continue
            if j.state.next_run_at_ms >= now:
                continue
            if j.schedule.kind in ("cron", "at"):
                missed.append(j)
            elif j.schedule.kind == "every":
                # Just reschedule from now instead of catching up
                j.state.next_run_at_ms = _compute_next_run(j.schedule, now)

        if missed:
            logger.info("Cron: {} missed job(s) detected, catching up", len(missed))
        for job in missed:
            logger.info("Cron: catching up missed job '{}' (was due {}ms ago)", job.name, now - job.state.next_run_at_ms)
            await self._execute_job(job)
        if missed:
            self._save_store()

    def _get_next_wake_ms(self) -> int | None:
        """Get the earliest next run time across all jobs."""
        if not self._store:
            return None
        times = [j.state.next_run_at_ms for j in self._store.jobs
                 if j.enabled and j.state.next_run_at_ms]
        return min(times) if times else None

    def _start_tick_loop(self) -> None:
        """Start the periodic tick loop."""
        if self._tick_task:
            self._tick_task.cancel()
        if not self._running:
            return
        self._tick_task = asyncio.create_task(self._tick_loop())

    async def _tick_loop(self) -> None:
        """Periodically check for due jobs. Sleeps at most _MAX_TICK_INTERVAL_S between checks."""
        while self._running:
            try:
                next_wake = self._get_next_wake_ms()
                if next_wake:
                    delay_s = max(0, (next_wake - _now_ms()) / 1000)
                    # Cap the sleep so we re-check frequently
                    sleep_s = min(delay_s, _MAX_TICK_INTERVAL_S)
                else:
                    sleep_s = _MAX_TICK_INTERVAL_S

                await asyncio.sleep(sleep_s)

                if not self._running:
                    break

                await self._on_tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cron: tick loop error: {}", e)
                # Avoid tight error loop
                await asyncio.sleep(5)

    async def _on_tick(self) -> None:
        """Handle tick - run due jobs."""
        self._load_store()
        if not self._store:
            return

        now = _now_ms()
        due_jobs = [
            j for j in self._store.jobs
            if j.enabled and j.state.next_run_at_ms and now >= j.state.next_run_at_ms
        ]

        for job in due_jobs:
            await self._execute_job(job)

        if due_jobs:
            self._save_store()

    async def _execute_job(self, job: CronJob) -> None:
        """Execute a single job."""
        start_ms = _now_ms()
        logger.info("Cron: executing job '{}' ({})", job.name, job.id)

        try:
            response = None
            if self.on_job:
                response = await self.on_job(job)

            job.state.last_status = "ok"
            job.state.last_error = None
            logger.info("Cron: job '{}' completed", job.name)
            _append_audit_log(self.store_path, job, "ok")

        except Exception as e:
            job.state.last_status = "error"
            job.state.last_error = str(e)
            logger.error("Cron: job '{}' failed: {}", job.name, e)
            _append_audit_log(self.store_path, job, "error", str(e))

        job.state.last_run_at_ms = start_ms
        job.updated_at_ms = _now_ms()

        # Handle one-shot jobs
        if job.schedule.kind == "at":
            if job.delete_after_run:
                self._store.jobs = [j for j in self._store.jobs if j.id != job.id]
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
        else:
            # Compute next run
            job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())

    # ========== Public API ==========

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        """List all jobs."""
        store = self._load_store()
        jobs = store.jobs if include_disabled else [j for j in store.jobs if j.enabled]
        return sorted(jobs, key=lambda j: j.state.next_run_at_ms or float('inf'))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        channel: str | None = None,
        to: str | None = None,
        delete_after_run: bool = False,
    ) -> CronJob:
        """Add a new job."""
        store = self._load_store()
        _validate_schedule_for_add(schedule)
        now = _now_ms()

        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=name,
            enabled=True,
            schedule=schedule,
            payload=CronPayload(
                kind="agent_turn",
                message=message,
                deliver=deliver,
                channel=channel,
                to=to,
            ),
            state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now)),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=delete_after_run,
        )

        store.jobs.append(job)
        self._save_store()

        logger.info("Cron: added job '{}' ({})", name, job.id)
        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a job by ID."""
        store = self._load_store()
        before = len(store.jobs)
        store.jobs = [j for j in store.jobs if j.id != job_id]
        removed = len(store.jobs) < before

        if removed:
            self._save_store()
            logger.info("Cron: removed job {}", job_id)

        return removed

    def enable_job(self, job_id: str, enabled: bool = True) -> CronJob | None:
        """Enable or disable a job."""
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                job.enabled = enabled
                job.updated_at_ms = _now_ms()
                if enabled:
                    job.state.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())
                else:
                    job.state.next_run_at_ms = None
                self._save_store()
                return job
        return None

    async def run_job(self, job_id: str, force: bool = False) -> bool:
        """Manually run a job."""
        store = self._load_store()
        for job in store.jobs:
            if job.id == job_id:
                if not force and not job.enabled:
                    return False
                await self._execute_job(job)
                self._save_store()
                return True
        return False

    def status(self) -> dict:
        """Get service status."""
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._get_next_wake_ms(),
        }
