from __future__ import annotations

import json

from click.testing import CliRunner
from test_robotics_contracts import task_dict

from cybernetics.cli.__main__ import main_cli


def test_top_level_help_lists_robot_task() -> None:
    result = CliRunner().invoke(main_cli, ["--help"])

    assert result.exit_code == 0
    assert "robot-task" in result.output


def test_robot_task_validate_json_reports_task_hash(tmp_path) -> None:
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task_dict()), encoding="utf-8")

    result = CliRunner().invoke(
        main_cli,
        ["--format", "json", "robot-task", "validate", str(task_path)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "valid"
    assert payload["task_id"] == "fixture_walk"
    assert len(payload["task_spec_hash"]) == 64


def test_robot_task_run_fixture_writes_artifacts(tmp_path) -> None:
    task_path = tmp_path / "task.json"
    out_dir = tmp_path / "run"
    task_path.write_text(json.dumps(task_dict()), encoding="utf-8")

    result = CliRunner().invoke(
        main_cli,
        [
            "--format",
            "json",
            "robot-task",
            "run-fixture",
            str(task_path),
            str(out_dir),
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "succeeded"
    assert payload["run_record_path"] == str(out_dir / "run_record.json")
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "rollout.json").exists()


def test_robot_task_policy_artifact_writes_lineage(tmp_path) -> None:
    task_path = tmp_path / "task.json"
    policy_path = tmp_path / "policy.json"
    task_path.write_text(json.dumps(task_dict()), encoding="utf-8")

    result = CliRunner().invoke(
        main_cli,
        [
            "--format",
            "json",
            "robot-task",
            "policy-artifact",
            str(task_path),
            str(policy_path),
            "--artifact-id",
            "pol_fixture",
            "--created-by-run-id",
            "rrun_fixture",
            "--checkpoint-uri",
            "worldlines://fixture/checkpoint",
            "--policy-format",
            "worldlines",
            "--rollout-artifact",
            "rollout.json",
            "--eval-metric",
            "success_rate=1.0",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["artifact_id"] == "pol_fixture"
    assert payload["policy_format"] == "worldlines"
    saved = json.loads(policy_path.read_text(encoding="utf-8"))
    assert saved["task_spec_hash"] == payload["task_spec_hash"]
    assert saved["eval_metrics"] == {"success_rate": 1.0}
