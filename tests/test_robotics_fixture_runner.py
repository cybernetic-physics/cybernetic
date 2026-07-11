from __future__ import annotations

import json

import pytest
from test_robotics_contracts import task_dict

from cybernetics.robotics import (
    FixtureRobotEnv,
    GymnasiumRobotEnvAdapter,
    LocoMuJoCoRobotEnv,
    RobotBackendError,
    RobotEnv,
    RobotRunRecord,
    RobotTaskSpec,
    run_robot_episode,
)


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


class FakeGymnasiumEnv:
    def __init__(self) -> None:
        self.position = 0.0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.position = float((options or {}).get("position", 0.0))
        return {"position": self.position, "seed": seed}, {"reset": True}

    def step(self, action):
        self.position += float(action)
        return (
            {"position": self.position},
            self.position,
            self.position >= 2.0,
            False,
            {"raw_action": action},
        )

    def render(self):
        return {"frame_position": self.position}

    def get_state(self):
        return {"position": self.position}

    def set_state(self, state):
        self.position = float(state["position"])

    def close(self) -> None:
        self.closed = True


def test_gymnasium_adapter_wraps_env_without_runtime_dependency() -> None:
    env = GymnasiumRobotEnvAdapter(FakeGymnasiumEnv(), action_key="delta")

    assert isinstance(env, RobotEnv)
    assert env.reset(seed=1) == {"position": 0.0, "seed": 1}
    first = env.step({"delta": 1.25})
    assert first.observation == {"position": 1.25}
    assert first.reward == 1.25
    assert first.terminated is False
    second = env.step({"delta": 0.75})
    assert second.terminated is True
    assert env.render() == {"frame_position": 2.0}
    assert env.capture({"mode": "rgb_array"})["backend_id"] == "gymnasium"
    env.set_state({"position": 0.5})
    assert env.get_state() == {"position": 0.5}
    env.close()
    assert env.closed is True


def test_locomujoco_adapter_can_wrap_existing_env_without_importing_runtime() -> None:
    task = RobotTaskSpec.from_dict(
        {**task_dict(), "simulator_backend": "locomujoco", "backend_config": {"image": "test"}}
    )

    env = LocoMuJoCoRobotEnv(task, env=FakeGymnasiumEnv(), action_key="delta")

    assert isinstance(env, RobotEnv)
    assert env.backend_id == "locomujoco"
    assert env.reset(seed=3)["seed"] == 3
    assert env.step({"delta": 2.0}).terminated is True
    env.close()


def test_locomujoco_adapter_missing_runtime_is_diagnosable(monkeypatch) -> None:
    def missing_runtime(**_kwargs):
        raise RobotBackendError("missing optional runtime")

    monkeypatch.setattr("cybernetics.robotics.locomujoco._make_locomujoco_env", missing_runtime)

    with pytest.raises(RobotBackendError, match="missing optional runtime"):
        LocoMuJoCoRobotEnv(env_name="UnitreeH1.run.real")
