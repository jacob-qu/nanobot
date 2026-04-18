"""Tests for cron model selection feature."""

from nanobot.cron.types import CronJob, CronPayload, CronSchedule


def test_cron_payload_model_defaults_to_none():
    payload = CronPayload(message="hello")
    assert payload.model is None


def test_cron_payload_accepts_model():
    payload = CronPayload(message="hello", model="deepseek/deepseek-chat")
    assert payload.model == "deepseek/deepseek-chat"


def test_cron_job_from_dict_preserves_model():
    data = {
        "id": "abc",
        "name": "test",
        "schedule": {"kind": "every", "every_ms": 60000},
        "payload": {"message": "hello", "model": "deepseek/deepseek-chat"},
        "state": {},
    }
    job = CronJob.from_dict(data)
    assert job.payload.model == "deepseek/deepseek-chat"


def test_cron_job_from_dict_without_model():
    data = {
        "id": "abc",
        "name": "test",
        "schedule": {"kind": "every", "every_ms": 60000},
        "payload": {"message": "hello"},
        "state": {},
    }
    job = CronJob.from_dict(data)
    assert job.payload.model is None
