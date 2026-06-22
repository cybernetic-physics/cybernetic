from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from cybernetics.cli.__main__ import main_cli
from cybernetics.cli.commands import doctor


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for var in (
        "CYBERNETICS_API_KEY",
        "CP_API_KEY",
        "WORLDLINES_API_KEY",
        "CYBERNETICS_BASE_URL",
        "CP_API_BASE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_top_level_help_lists_doctor() -> None:
    result = CliRunner().invoke(main_cli, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.output


def test_doctor_json_reports_luc_sft_only_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CYBERNETICS_API_KEY", "cp_live_test")
    monkeypatch.setenv("CYBERNETICS_BASE_URL", "https://luc-api.cyberneticphysics.com")
    monkeypatch.setattr(
        doctor,
        "_fetch_health",
        lambda base_url: doctor._HealthCheck(ok=True, path="/health", status_code=200),
    )
    monkeypatch.setattr(
        doctor,
        "_fetch_capabilities",
        lambda base_url, api_key: {
            "supports_training": True,
            "supports_sampling": True,
            "loss_families": ["cross_entropy"],
            "dreamzero_rl_available": False,
            "dreamzero_rl_unavailable_reason": "DreamZero RL requires `groot.vla.rl`.",
        },
    )

    result = CliRunner().invoke(main_cli, ["--format", "json", "doctor"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["base_url"] == "https://luc-api.cyberneticphysics.com"
    assert payload["auth_source"] == "CYBERNETICS_API_KEY"
    assert payload["dreamzero_sft_ready"] is True
    assert payload["dreamzero_rl_ready"] is False
    assert "groot.vla.rl" in payload["dreamzero_rl_unavailable_reason"]
    assert "cp_live_test" not in result.output


def test_doctor_accepts_cp_env_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CP_API_KEY", "cp_live_alias")
    monkeypatch.setenv("CP_API_BASE", "https://luc-api.cyberneticphysics.com")
    monkeypatch.setattr(
        doctor,
        "_fetch_health",
        lambda base_url: doctor._HealthCheck(ok=True, path="/health", status_code=200),
    )
    monkeypatch.setattr(
        doctor,
        "_fetch_capabilities",
        lambda base_url, api_key: {
            "supports_training": True,
            "supports_sampling": True,
            "loss_families": ["cross_entropy", "flow_rwr"],
            "dreamzero_rl_available": True,
        },
    )

    result = CliRunner().invoke(main_cli, ["--format", "json", "doctor", "--require-rl"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["base_url"] == "https://luc-api.cyberneticphysics.com"
    assert payload["auth_source"] == "CP_API_KEY"
    assert payload["dreamzero_rl_ready"] is True
    assert "cp_live_alias" not in result.output


def test_doctor_require_rl_exits_nonzero_when_rl_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CYBERNETICS_API_KEY", "cp_live_test")
    monkeypatch.setattr(
        doctor,
        "_fetch_health",
        lambda base_url: doctor._HealthCheck(ok=True, path="/health", status_code=200),
    )
    monkeypatch.setattr(
        doctor,
        "_fetch_capabilities",
        lambda base_url, api_key: {
            "supports_training": True,
            "loss_families": ["cross_entropy"],
            "dreamzero_rl_available": False,
        },
    )

    result = CliRunner().invoke(main_cli, ["--format", "json", "doctor", "--require-rl"])

    assert result.exit_code == 1
    assert json.loads(result.output)["dreamzero_rl_ready"] is False


def test_doctor_without_key_errors() -> None:
    result = CliRunner().invoke(main_cli, ["doctor"])
    assert result.exit_code != 0
    assert "No API key found" in str(result.exception)


def test_summarize_capabilities_marks_rl_ready_only_when_loss_is_advertised() -> None:
    readiness = doctor._summarize_capabilities(
        {
            "supports_training": True,
            "supports_sampling": True,
            "loss_families": ["cross_entropy", "flow_rwr"],
            "dreamzero_rl_available": True,
        },
        "flow_rwr",
    )

    assert readiness.dreamzero_sft_ready is True
    assert readiness.dreamzero_rl_ready is True
