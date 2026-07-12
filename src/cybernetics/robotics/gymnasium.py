"""Dependency-light adapters for already-constructed Gymnasium environments."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .env import StepResult, VectorStepResult


class RobotBackendError(RuntimeError):
    """An environment backend violated the robotics runtime contract."""


class GymnasiumRobotEnvAdapter:
    """Wrap one Gymnasium-shaped environment behind :class:`RobotEnv`.

    The adapter never imports Gymnasium. Runtime images construct the native
    environment and pass it in, keeping simulator dependencies out of the SDK.
    """

    backend_id = "gymnasium"

    def __init__(
        self,
        env: Any,
        *,
        backend_id: Optional[str] = None,
        action_key: Optional[str] = None,
    ) -> None:
        self.env = env
        self.backend_id = backend_id or self.backend_id
        self.action_key = action_key
        self.closed = False
        self.last_info: Mapping[str, Any] = {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        raw = self.env.reset(seed=seed, options=dict(options or {}))
        observation, info = _split_reset(raw)
        self.last_info = _as_info(info)
        return _as_observation(observation)

    def step(self, action: Mapping[str, Any]) -> StepResult:
        raw = self.env.step(_unwrap_action(action, self.action_key))
        if not isinstance(raw, tuple) or len(raw) not in (4, 5):
            raise RobotBackendError("Gymnasium step() must return a 4- or 5-item tuple")
        if len(raw) == 5:
            observation, reward, terminated, truncated, info = raw
        else:
            observation, reward, done, info = raw
            terminated, truncated = _split_legacy_done(done, info)
        self.last_info = _as_info(info)
        return StepResult(
            observation=_as_observation(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=self.last_info,
        )

    def render(self, mode: str = "rgb_array") -> Any:
        del mode
        render = getattr(self.env, "render", None)
        if not callable(render):
            raise RobotBackendError("Gymnasium env does not expose render()")
        return render()

    def capture(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        mode = str(request.get("mode", "rgb_array"))
        return {
            "kind": "gymnasium_render",
            "backend_id": self.backend_id,
            "mode": mode,
            "frame": self.render(mode),
        }

    def get_state(self) -> Mapping[str, Any]:
        get_state = getattr(self.env, "get_state", None)
        if callable(get_state):
            return _as_observation(get_state())
        state = getattr(self.env, "state", None)
        if isinstance(state, Mapping):
            return dict(state)
        raise RobotBackendError("Gymnasium env cannot snapshot state")

    def set_state(self, state: Mapping[str, Any]) -> None:
        set_state = getattr(self.env, "set_state", None)
        if callable(set_state):
            set_state(dict(state))
            return
        if hasattr(self.env, "state"):
            setattr(self.env, "state", dict(state))
            return
        raise RobotBackendError("Gymnasium env cannot restore state")

    def close(self) -> None:
        if self.closed:
            return
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        self.closed = True


class GymnasiumVectorEnvAdapter:
    """Wrap a Gymnasium VectorEnv without copying batched CPU/GPU values."""

    backend_id = "gymnasium-vector"

    def __init__(
        self,
        env: Any,
        *,
        backend_id: Optional[str] = None,
        action_key: Optional[str] = None,
    ) -> None:
        self.env = env
        self.backend_id = backend_id or self.backend_id
        self.action_key = action_key
        self.num_envs = int(getattr(env, "num_envs", 0))
        if self.num_envs <= 0:
            raise RobotBackendError("vector env must expose a positive num_envs")
        self.closed = False
        self.last_info: Any = {}

    def reset(
        self,
        seed: Optional[int | Sequence[int]] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        raw = self.env.reset(seed=seed, options=dict(options or {}))
        observation, info = _split_reset(raw)
        self.last_info = info if info is not None else {}
        return observation

    def step(self, actions: Any) -> VectorStepResult:
        raw = self.env.step(_unwrap_vector_action(actions, self.action_key))
        if not isinstance(raw, tuple) or len(raw) not in (4, 5):
            raise RobotBackendError("Gymnasium vector step() must return a 4- or 5-item tuple")
        if len(raw) == 5:
            observations, rewards, terminated, truncated, info = raw
        else:
            observations, rewards, done, info = raw
            terminated, truncated = _split_legacy_done(done, info)
        for name, value in (
            ("rewards", rewards),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            _validate_batch_width(name, value, self.num_envs)
        self.last_info = info if info is not None else {}
        return VectorStepResult(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=self.last_info,
            num_envs=self.num_envs,
        )

    def render(self, mode: str = "rgb_array") -> Any:
        del mode
        render = getattr(self.env, "render", None)
        if not callable(render):
            raise RobotBackendError("Gymnasium vector env does not expose render()")
        return render()

    def capture(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        mode = str(request.get("mode", "rgb_array"))
        return {
            "kind": "gymnasium_vector_render",
            "backend_id": self.backend_id,
            "mode": mode,
            "frames": self.render(mode),
        }

    def get_state(self) -> Any:
        get_state = getattr(self.env, "get_state", None)
        if callable(get_state):
            return get_state()
        raise RobotBackendError("Gymnasium vector env cannot snapshot state")

    def set_state(self, state: Any) -> None:
        set_state = getattr(self.env, "set_state", None)
        if not callable(set_state):
            raise RobotBackendError("Gymnasium vector env cannot restore state")
        set_state(state)

    def close(self) -> None:
        if self.closed:
            return
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        self.closed = True


def _split_reset(raw: Any) -> tuple[Any, Any]:
    if isinstance(raw, tuple) and len(raw) == 2:
        return raw
    return raw, {}


def _as_observation(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {"observation": value}


def _as_info(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"info": value}


def _unwrap_action(action: Mapping[str, Any], action_key: Optional[str]) -> Any:
    if action_key is None:
        return dict(action)
    if action_key not in action:
        raise RobotBackendError(f"action is missing configured action_key {action_key!r}")
    return action[action_key]


def _unwrap_vector_action(actions: Any, action_key: Optional[str]) -> Any:
    if action_key is None:
        return actions
    if not isinstance(actions, Mapping) or action_key not in actions:
        raise RobotBackendError(f"action batch is missing configured action_key {action_key!r}")
    return actions[action_key]


def _split_legacy_done(done: Any, info: Any) -> tuple[Any, Any]:
    truncated = info.get("TimeLimit.truncated", False) if isinstance(info, Mapping) else False
    return _mask_and(done, _mask_not(truncated)), truncated


def _mask_not(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return type(value)(not bool(item) for item in value)
    if isinstance(value, bool):
        return not value
    try:
        return ~value
    except TypeError as exc:
        raise RobotBackendError("legacy done mask cannot be inverted") from exc


def _mask_and(left: Any, right: Any) -> Any:
    if isinstance(left, (list, tuple)):
        if isinstance(right, (list, tuple)):
            if len(left) != len(right):
                raise RobotBackendError("legacy done and truncation masks have different widths")
            return type(left)(bool(a) and bool(b) for a, b in zip(left, right, strict=True))
        return type(left)(bool(item) and bool(right) for item in left)
    if isinstance(left, bool) and isinstance(right, bool):
        return left and right
    try:
        return left & right
    except TypeError as exc:
        raise RobotBackendError("legacy done masks cannot be combined") from exc


def _validate_batch_width(name: str, value: Any, expected: int) -> None:
    if expected == 1 and isinstance(value, (bool, int, float)):
        return
    try:
        actual = len(value)
    except TypeError as exc:
        raise RobotBackendError(f"vector {name} does not expose a batch dimension") from exc
    if actual != expected:
        raise RobotBackendError(f"vector {name} width {actual} does not match num_envs {expected}")
