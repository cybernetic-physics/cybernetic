"""Tiny RobotEnv runner skeleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .artifacts import write_json_artifact, write_robot_run_record
from .contracts import ROBOT_RUN_SCHEMA_VERSION, RobotRunRecord, RobotTaskSpec, stable_hash
from .env import RobotEnv, StepResult

ActionFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def default_action(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"delta": 1.0}


def deterministic_run_id(task_spec: RobotTaskSpec, seed: int) -> str:
    digest = stable_hash({"seed": seed, "task_spec_hash": task_spec.task_hash()})
    return f"rrun_{digest[:16]}"


def run_robot_episode(
    task_spec: RobotTaskSpec,
    env: RobotEnv,
    out_dir: str | Path,
    *,
    seed: int = 0,
    action_fn: Optional[ActionFn] = None,
    max_steps: Optional[int] = None,
) -> RobotRunRecord:
    """Run one deterministic episode and write run artifacts.

    This is intentionally small: it exists to prove the contract boundary before
    any MuJoCo, LocoMuJoCo, Isaac, or Worldlines adapter is wired in.
    """

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_hash = task_spec.task_hash()
    run_id = deterministic_run_id(task_spec, seed)
    limit = int(max_steps or task_spec.eval_protocol.get("max_steps", 32))
    action = action_fn or default_action
    rollout: list[dict[str, Any]] = []
    status = "failed"
    error: str | None = None
    total_reward = 0.0

    try:
        observation = env.reset(seed=seed, options=task_spec.reset_spec)
        for step_index in range(limit):
            result: StepResult = env.step(action(observation))
            total_reward += float(result.reward)
            rollout.append({"step": step_index, **result.to_dict()})
            observation = result.observation
            if result.terminated:
                status = "succeeded"
                break
            if result.truncated:
                status = "truncated"
                break
        else:
            status = "truncated"
    except Exception as exc:  # pragma: no cover - tested through behavior, not branch coverage
        error = f"{type(exc).__name__}: {exc}"
        status = "failed"
    finally:
        env.close()

    metrics = {
        "run_id": run_id,
        "task_spec_hash": task_hash,
        "status": status,
        "steps": len(rollout),
        "total_reward": total_reward,
    }
    write_json_artifact(output / "rollout.json", {"run_id": run_id, "steps": rollout})
    write_json_artifact(output / "metrics.json", metrics)

    record = RobotRunRecord(
        schema_version=ROBOT_RUN_SCHEMA_VERSION,
        run_id=run_id,
        task_spec_uri=f"task://{task_spec.task_id}",
        task_spec_hash=task_hash,
        backend_image=str(task_spec.backend_config.get("image", task_spec.simulator_backend)),
        seed=int(seed),
        status=status,
        logs_uri=str(output / "rollout.json"),
        metrics_uri=str(output / "metrics.json"),
        artifacts_uri=str(output),
        error=error,
    )
    write_robot_run_record(output / "run_record.json", record)
    return record
