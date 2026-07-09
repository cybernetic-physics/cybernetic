from __future__ import annotations

import json

from click.testing import CliRunner

from cybernetics.cli.__main__ import main_cli


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
