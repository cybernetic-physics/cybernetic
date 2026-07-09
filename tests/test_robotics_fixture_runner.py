from __future__ import annotations

import json

from cybernetics.robotics import FixtureRobotEnv, RobotRunRecord, RobotTaskSpec, run_robot_episode

from test_robotics_contracts import task_dict


def test_fixture_robot_env_reset_step_state_and_close() -> None:
    env = FixtureRobotEnv(max_steps=4, success_position=2.0)
    obs = env.reset(seed=7)

    assert obs == {"position": 0.0, "step_count": 0}
    first = env.step({"delta": 1.25})
    assert first.observation["position"] == 1.25
    assert first.terminated is False
    second = env.step({"delta": 0.75})
    assert second.terminated is True
    assert env.get_state()["step_count"] == 2
    env.close()


def test_run_robot_episode_writes_deterministic_artifacts(tmp_path) -> None:
    spec = RobotTaskSpec.from_dict(task_dict())

    first = run_robot_episode(spec, FixtureRobotEnv(), tmp_path / "a", seed=42)
    second = run_robot_episode(spec, FixtureRobotEnv(), tmp_path / "b", seed=42)

    assert first.run_id == second.run_id
    assert first.status == "succeeded"
    assert first.task_spec_hash == spec.task_hash()

    run_record_path = tmp_path / "a" / "run_record.json"
    metrics_path = tmp_path / "a" / "metrics.json"
    rollout_path = tmp_path / "a" / "rollout.json"
    assert run_record_path.exists()
    assert metrics_path.exists()
    assert rollout_path.exists()

    run_record = RobotRunRecord.from_dict(json.loads(run_record_path.read_text()))
    assert run_record.run_id == first.run_id
    assert run_record.task_spec_hash == spec.task_hash()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["task_spec_hash"] == spec.task_hash()
    assert metrics["status"] == "succeeded"


def test_run_robot_episode_failure_is_diagnosable(tmp_path) -> None:
    spec = RobotTaskSpec.from_dict(task_dict())

    def bad_action(_obs):
        raise RuntimeError("policy exploded")

    record = run_robot_episode(spec, FixtureRobotEnv(), tmp_path, seed=1, action_fn=bad_action)

    assert record.status == "failed"
    assert record.error is not None
    assert "policy exploded" in record.error
    saved = RobotRunRecord.from_dict(json.loads((tmp_path / "run_record.json").read_text()))
    assert saved.status == "failed"
    assert "policy exploded" in (saved.error or "")
