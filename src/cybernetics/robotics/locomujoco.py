"""Optional Gymnasium/LocoMuJoCo RobotEnv adapters.

The base SDK must import without Gymnasium, MuJoCo, or LocoMuJoCo installed.
This module therefore exposes a real wrapper for already-created
Gymnasium-shaped envs and lazy construction for LocoMuJoCo when optional runtime
packages are present.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .contracts import RobotTaskSpec
from .env import StepResult


class RobotBackendError(RuntimeError):
    """A simulator backend could not be constructed or used."""


class GymnasiumRobotEnvAdapter:
    """Wrap a Gymnasium-shaped env behind the RobotEnv protocol."""

    backend_id = "gymnasium"

    def __init__(
        self,
        env: Any,
        *,
        task_spec: Optional[RobotTaskSpec] = None,
        backend_id: str | None = None,
        action_key: str | None = None,
        backend_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.task_spec = task_spec
        self.backend_id = backend_id or self.backend_id
        self.action_key = action_key
        self.backend_config = dict(backend_config or {})
        self.closed = False
        self._last_observation: Mapping[str, Any] | None = None
        self._last_info: Mapping[str, Any] = {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        raw = self.env.reset(seed=seed, options=dict(options or {}))
        observation, info = _split_reset(raw)
        self._last_observation = _as_observation_mapping(observation)
        self._last_info = _as_info_mapping(info)
        return self._last_observation

    def step(self, action: Mapping[str, Any]) -> StepResult:
        raw = self.env.step(_gym_action(action, action_key=self.action_key))
        if not isinstance(raw, tuple) or len(raw) not in (4, 5):
            raise RobotBackendError("Gymnasium env step() must return a 4- or 5-item tuple")
        if len(raw) == 5:
            observation, reward, terminated, truncated, info = raw
        else:
            observation, reward, done, info = raw
            terminated = bool(done)
            truncated = False
        mapped_observation = _as_observation_mapping(observation)
        self._last_observation = mapped_observation
        self._last_info = _as_info_mapping(info)
        return StepResult(
            observation=mapped_observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=self._last_info,
        )

    def render(self, mode: str = "rgb_array") -> Any:
        try:
            return self.env.render()
        except TypeError:
            return self.env.render(mode=mode)

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
            return _as_observation_mapping(get_state())
        state = getattr(self.env, "state", None)
        if isinstance(state, Mapping):
            return dict(state)
        raise RobotBackendError(
            "Gymnasium env does not expose get_state(); backend cannot snapshot state"
        )

    def set_state(self, state: Mapping[str, Any]) -> None:
        set_state = getattr(self.env, "set_state", None)
        if callable(set_state):
            set_state(dict(state))
            return
        if hasattr(self.env, "state"):
            setattr(self.env, "state", dict(state))
            return
        raise RobotBackendError(
            "Gymnasium env does not expose set_state(); backend cannot restore state"
        )

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        self.closed = True


class LocoMuJoCoRobotEnv(GymnasiumRobotEnvAdapter):
    """LocoMuJoCo/MuJoCo adapter with lazy optional runtime imports."""

    backend_id = "locomujoco"

    def __init__(
        self,
        task_spec: Optional[RobotTaskSpec] = None,
        *,
        env: Any | None = None,
        env_id: str | None = None,
        env_name: str | None = None,
        render_mode: str | None = None,
        action_key: str | None = None,
        **backend_config: Any,
    ) -> None:
        merged_config = _merged_backend_config(task_spec, backend_config)
        resolved_action_key = action_key or _optional_str(merged_config.get("action_key"))
        if env is None:
            env = _make_locomujoco_env(
                env_id=env_id or _optional_str(merged_config.get("env_id")) or "LocoMujoco",
                env_name=env_name or _optional_str(merged_config.get("env_name")),
                render_mode=render_mode or _optional_str(merged_config.get("render_mode")),
                gym_kwargs=_mapping_value(merged_config.get("gym_kwargs")),
            )
        super().__init__(
            env,
            task_spec=task_spec,
            backend_id=self.backend_id,
            action_key=resolved_action_key,
            backend_config=merged_config,
        )


def _make_locomujoco_env(
    *,
    env_id: str,
    env_name: str | None,
    render_mode: str | None,
    gym_kwargs: Mapping[str, Any],
) -> Any:
    try:
        import gymnasium as gym
        import loco_mujoco  # noqa: F401
    except ImportError as exc:
        raise RobotBackendError(
            "LocoMuJoCoRobotEnv requires optional runtime packages. Install "
            "`gymnasium` and `loco-mujoco`, or pass an already-created env=."
        ) from exc

    kwargs = dict(gym_kwargs)
    if env_name is not None:
        kwargs.setdefault("env_name", env_name)
    if render_mode is not None:
        kwargs.setdefault("render_mode", render_mode)
    return gym.make(env_id, **kwargs)


def _merged_backend_config(
    task_spec: RobotTaskSpec | None,
    backend_config: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if task_spec is not None:
        merged.update(task_spec.backend_config)
    merged.update(dict(backend_config))
    return merged


def _split_reset(raw: Any) -> tuple[Any, Any]:
    if isinstance(raw, tuple) and len(raw) == 2:
        return raw
    return raw, {}


def _as_observation_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {"observation": value}


def _as_info_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"info": value}


def _gym_action(action: Mapping[str, Any], *, action_key: str | None) -> Any:
    if action_key is not None:
        if action_key not in action:
            raise RobotBackendError(f"action mapping is missing configured action_key {action_key!r}")
        return action[action_key]
    if len(action) == 1:
        return next(iter(action.values()))
    return dict(action)


def _mapping_value(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RobotBackendError("backend_config.gym_kwargs must be an object")
    return dict(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
