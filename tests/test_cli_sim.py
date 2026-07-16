from __future__ import annotations

import json

import httpx
import respx
from click.testing import CliRunner

from cybernetics.cli.__main__ import main_cli

BASE = "https://api.test"


def test_top_level_help_lists_sim() -> None:
    result = CliRunner().invoke(main_cli, ["--help"])
    assert result.exit_code == 0
    assert "sim" in result.output


def test_sim_inspect_json_reports_usd_asset(tmp_path) -> None:
    (tmp_path / "scene.usd").write_text("#usda 1.0\n")

    result = CliRunner().invoke(main_cli, ["--format", "json", "sim", "inspect", str(tmp_path)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["root_relpath"] == "scene.usd"
    assert payload["asset_kind"] == "usd_stage"
    assert payload["compatibility_status"] == "ready_to_render"


def test_sim_render_public_flag_is_explicitly_not_mvp(tmp_path) -> None:
    (tmp_path / "scene.usd").write_text("#usda 1.0\n")

    result = CliRunner().invoke(main_cli, ["sim", "render", str(tmp_path), "--public"])

    assert result.exit_code != 0
    assert "Public /sim artifact pages are not implemented" in str(result.exception)


@respx.mock
def test_sim_launch_wait_reports_created_session_and_cleanup() -> None:
    respx.post(f"{BASE}/v1/sessions").mock(
        return_value=httpx.Response(
            200,
            json={"sessionId": "sess_demo", "status": "queued"},
        )
    )
    respx.get(f"{BASE}/v1/sessions/sess_demo").mock(
        return_value=httpx.Response(
            200,
            json={"sessionId": "sess_demo", "status": "queued"},
        )
    )
    respx.post(f"{BASE}/v1/sessions/sess_demo/stop").mock(return_value=httpx.Response(204))

    result = CliRunner().invoke(
        main_cli,
        [
            "sim",
            "launch",
            "cybernetics://envs/env_demo/versions/ver_demo",
            "--wait",
            "--timeout-seconds",
            "0",
            "--api-key",
            "cp_live_test",
            "--base-url",
            BASE,
        ],
    )

    assert result.exit_code != 0
    assert result.exception is not None
    assert "sess_demo" in str(result.exception)
    assert "automatic stop requested" in str(result.exception)
