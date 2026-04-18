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


from nanobot.config.schema import AgentDefaults, CronConfig


def test_cron_config_model_override_defaults_to_none():
    cfg = CronConfig()
    assert cfg.model_override is None


def test_cron_config_model_override_accepts_value():
    cfg = CronConfig(model_override="deepseek/deepseek-chat")
    assert cfg.model_override == "deepseek/deepseek-chat"


def test_cron_config_accepts_camel_case():
    cfg = CronConfig(**{"modelOverride": "deepseek/deepseek-chat"})
    assert cfg.model_override == "deepseek/deepseek-chat"


def test_agent_defaults_has_cron_config():
    defaults = AgentDefaults()
    assert isinstance(defaults.cron, CronConfig)
    assert defaults.cron.model_override is None


from nanobot.cron.service import CronService


def test_add_job_with_model(tmp_path):
    service = CronService(tmp_path / "cron" / "jobs.json")
    job = service.add_job(
        name="test",
        schedule=CronSchedule(kind="every", every_ms=60000),
        message="hello",
        model="deepseek/deepseek-chat",
    )
    assert job.payload.model == "deepseek/deepseek-chat"


def test_add_job_without_model(tmp_path):
    service = CronService(tmp_path / "cron" / "jobs.json")
    job = service.add_job(
        name="test",
        schedule=CronSchedule(kind="every", every_ms=60000),
        message="hello",
    )
    assert job.payload.model is None


from nanobot.agent.tools.cron import CronTool


def _make_cron_tool(tmp_path) -> CronTool:
    svc = CronService(tmp_path / "cron" / "jobs.json")
    tool = CronTool(svc, default_timezone="UTC")
    tool.set_context("telegram", "chat-1")
    return tool


def test_tool_add_job_with_model(tmp_path):
    tool = _make_cron_tool(tmp_path)
    result = tool._add_job(None, "test task", 60, None, None, None, model="deepseek/deepseek-chat")
    assert result.startswith("Created job")
    job = tool._cron.list_jobs()[0]
    assert job.payload.model == "deepseek/deepseek-chat"


def test_tool_add_job_without_model(tmp_path):
    tool = _make_cron_tool(tmp_path)
    result = tool._add_job(None, "test task", 60, None, None, None)
    assert result.startswith("Created job")
    job = tool._cron.list_jobs()[0]
    assert job.payload.model is None


def test_tool_list_shows_model(tmp_path):
    tool = _make_cron_tool(tmp_path)
    tool._cron.add_job(
        name="with-model",
        schedule=CronSchedule(kind="every", every_ms=60000),
        message="hello",
        model="deepseek/deepseek-chat",
    )
    result = tool._list_jobs()
    assert "Model: deepseek/deepseek-chat" in result


def test_tool_list_hides_model_when_none(tmp_path):
    tool = _make_cron_tool(tmp_path)
    tool._cron.add_job(
        name="no-model",
        schedule=CronSchedule(kind="every", every_ms=60000),
        message="hello",
    )
    result = tool._list_jobs()
    assert "Model:" not in result


def test_tool_schema_has_model_parameter():
    service_stub = type("S", (), {
        "list_jobs": lambda self: [],
        "get_job": lambda self, _: None,
        "remove_job": lambda self, _: "not-found",
        "add_job": lambda self, **kw: type("J", (), {"id": "x", "name": "x"})(),
    })()
    tool = CronTool(service_stub)
    assert "model" in tool.parameters["properties"]
