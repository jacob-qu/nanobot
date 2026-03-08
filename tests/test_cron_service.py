import asyncio
import json
import time

import pytest

from nanobot.cron.service import CronService, _now_ms
from nanobot.cron.types import CronJobState, CronSchedule


def test_add_job_rejects_unknown_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    with pytest.raises(ValueError, match="unknown timezone 'America/Vancovuer'"):
        service.add_job(
            name="tz typo",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancovuer"),
            message="hello",
        )

    assert service.list_jobs(include_disabled=True) == []


def test_add_job_accepts_valid_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    job = service.add_job(
        name="tz ok",
        schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancouver"),
        message="hello",
    )

    assert job.schedule.tz == "America/Vancouver"
    assert job.state.next_run_at_ms is not None


@pytest.mark.asyncio
async def test_running_service_honors_external_disable(tmp_path) -> None:
    store_path = tmp_path / "cron" / "jobs.json"
    called: list[str] = []

    async def on_job(job) -> None:
        called.append(job.id)

    service = CronService(store_path, on_job=on_job)
    job = service.add_job(
        name="external-disable",
        schedule=CronSchedule(kind="every", every_ms=2000),
        message="hello",
    )
    await service.start()
    try:
        external = CronService(store_path)
        updated = external.enable_job(job.id, enabled=False)
        assert updated is not None
        assert updated.enabled is False

        await asyncio.sleep(0.35)
        assert called == []
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_catch_up_missed_jobs_on_start(tmp_path) -> None:
    """Cron-type jobs whose next_run_at_ms is in the past should be executed on startup."""
    store_path = tmp_path / "cron" / "jobs.json"
    called: list[str] = []

    async def on_job(job) -> None:
        called.append(job.id)

    # Create a cron job and manually set its next_run_at_ms to the past
    service = CronService(store_path)
    job = service.add_job(
        name="missed-job",
        schedule=CronSchedule(kind="cron", expr="0 9 * * *"),
        message="hello",
    )
    # Tamper with the stored state to simulate a missed run
    data = json.loads(store_path.read_text(encoding="utf-8"))
    data["jobs"][0]["state"]["nextRunAtMs"] = _now_ms() - 60_000  # 60s ago
    store_path.write_text(json.dumps(data), encoding="utf-8")

    # Start a new service instance — it should catch up the missed job
    service2 = CronService(store_path, on_job=on_job)
    await service2.start()
    try:
        assert job.id in called
    finally:
        await service2.stop()


@pytest.mark.asyncio
async def test_audit_log_written_on_execution(tmp_path) -> None:
    """Each job execution should append a line to audit.log."""
    store_path = tmp_path / "cron" / "jobs.json"

    async def on_job(job) -> None:
        pass

    service = CronService(store_path, on_job=on_job)
    job = service.add_job(
        name="audit-test",
        schedule=CronSchedule(kind="every", every_ms=100),
        message="hello",
    )
    await service.start()
    try:
        await asyncio.sleep(0.25)
    finally:
        await service.stop()

    audit_path = tmp_path / "cron" / "audit.log"
    assert audit_path.exists()
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    assert "audit-test" in lines[0]
    assert "ok" in lines[0]


@pytest.mark.asyncio
async def test_tick_loop_recovers_from_job_error(tmp_path) -> None:
    """The tick loop should continue running even if a job raises an exception."""
    store_path = tmp_path / "cron" / "jobs.json"
    call_count = 0

    async def on_job(job) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")

    service = CronService(store_path, on_job=on_job)
    service.add_job(
        name="error-recovery",
        schedule=CronSchedule(kind="every", every_ms=100),
        message="hello",
    )
    await service.start()
    try:
        await asyncio.sleep(0.35)
    finally:
        await service.stop()

    # Should have been called more than once despite the first error
    assert call_count >= 2

    # Audit log should contain both the error and a subsequent ok
    audit_path = tmp_path / "cron" / "audit.log"
    content = audit_path.read_text(encoding="utf-8")
    assert "error" in content
    assert "ok" in content
