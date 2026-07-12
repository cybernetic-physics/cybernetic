from __future__ import annotations

import pytest

from cybernetics.robotics import (
    GymnasiumRobotEnvAdapter,
    GymnasiumVectorEnvAdapter,
    RobotBackendError,
    RobotEnv,
    VectorRobotEnv,
)


class FakeEnv:
    def __init__(self) -> None:
        self.position = 0.0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.position = float((options or {}).get("position", 0.0))
        return {"position": self.position, "seed": seed}, {"reset": True}

    def step(self, action):
        self.position += float(action)
        return {"position": self.position}, self.position, self.position >= 2, False, {}

    def render(self):
        return {"position": self.position}

    def get_state(self):
        return {"position": self.position}

    def set_state(self, state):
        self.position = float(state["position"])

    def close(self):
        self.closed = True


class FakeVectorEnv:
    num_envs = 2

    def __init__(self, *, bad_width: bool = False) -> None:
        self.bad_width = bad_width
        self.closed = False

    def reset(self, *, seed=None, options=None):
        return {"position": [0.0, 0.0]}, {"seed": seed, "options": options}

    def step(self, actions):
        rewards = [1.0] if self.bad_width else [1.0, 2.0]
        return {"position": actions}, rewards, [False, True], [False, False], {}

    def render(self):
        return [b"frame-a", b"frame-b"]

    def get_state(self):
        return [{"position": 0.0}, {"position": 0.0}]

    def set_state(self, state):
        self.state = state

    def close(self):
        self.closed = True


def test_single_adapter_preserves_robot_env_contract() -> None:
    native = FakeEnv()
    env = GymnasiumRobotEnvAdapter(native, action_key="delta")

    assert isinstance(env, RobotEnv)
    assert env.reset(seed=7, options={"position": 0.5})["position"] == 0.5
    assert env.step({"delta": 1.5}).terminated is True
    env.set_state({"position": 0.25})
    assert env.get_state() == {"position": 0.25}
    env.close()
    env.close()
    assert native.closed is True


def test_vector_adapter_preserves_batch_values() -> None:
    native = FakeVectorEnv()
    env = GymnasiumVectorEnvAdapter(native)

    assert isinstance(env, VectorRobotEnv)
    assert env.reset(seed=[1, 2])["position"] == [0.0, 0.0]
    result = env.step([[1], [2]])
    assert result.num_envs == 2
    assert result.rewards == [1.0, 2.0]
    assert result.terminated == [False, True]
    assert env.capture({})["frames"] == [b"frame-a", b"frame-b"]
    env.close()
    assert native.closed is True


def test_vector_adapter_rejects_wrong_batch_width() -> None:
    env = GymnasiumVectorEnvAdapter(FakeVectorEnv(bad_width=True))

    with pytest.raises(RobotBackendError, match="rewards width"):
        env.step([[1], [2]])
