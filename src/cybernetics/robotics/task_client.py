"""User-facing RobotTask SDK facade.

The lower-level modules own contracts, environments, and artifact writers. This
facade owns the ergonomic SDK boundary: validate a task, run a local env, and
produce artifacts without importing simulator runtimes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .artifacts import write_json_artifact, write_policy_artifact
from .contracts import ROBOT_POLICY_SCHEMA_VERSION, PolicyArtifact, RobotTaskSpec
from .env import RobotEnv
from .fixture import FixtureRobotEnv
from .runner import ActionFn, run_robot_episode

TaskSpecInput = RobotTaskSpec | Mapping[str, Any]


@dataclass(frozen=True)
class RobotTaskRunResult:
    """Paths and records produced by a local RobotTask run."""

    task_spec: RobotTaskSpec
    run_record_path: Path
    metrics_path: Path
    rollout_path: Path
    output_dir: Path

    @property
    def run_record(self):
        from .contracts import RobotRunRecord

        return RobotRunRecord.from_dict(json.loads(self.run_record_path.read_text()))

    def to_dict(self) -> dict[str, Any]:
        record = self.run_record
        return {
            "task_id": self.task_spec.task_id,
            "task_spec_hash": self.task_spec.task_hash(),
            "run_id": record.run_id,
            "status": record.status,
            "output_dir": str(self.output_dir),
            "run_record_path": str(self.run_record_path),
            "metrics_path": str(self.metrics_path),
            "rollout_path": str(self.rollout_path),
        }


class RobotTasksClient:
    """Small composition surface for RobotTask authoring and local validation."""

    def validate(self, task_spec: TaskSpecInput) -> RobotTaskSpec:
        if isinstance(task_spec, RobotTaskSpec):
            return RobotTaskSpec.from_dict(task_spec.to_dict())
        return RobotTaskSpec.from_dict(task_spec)

    def load(self, path: str | Path) -> RobotTaskSpec:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("RobotTaskSpec file must contain a JSON object")
        return self.validate(payload)

    def save(self, task_spec: TaskSpecInput, path: str | Path) -> Path:
        return write_json_artifact(path, self.validate(task_spec).to_dict())

    def run_local(
        self,
        task_spec: TaskSpecInput,
        env: RobotEnv,
        out_dir: str | Path,
        *,
        seed: int = 0,
        action_fn: Optional[ActionFn] = None,
        max_steps: Optional[int] = None,
    ) -> RobotTaskRunResult:
        spec = self.validate(task_spec)
        output = Path(out_dir)
        run_robot_episode(
            spec,
            env,
            output,
            seed=seed,
            action_fn=action_fn,
            max_steps=max_steps,
        )
        return RobotTaskRunResult(
            task_spec=spec,
            run_record_path=output / "run_record.json",
            metrics_path=output / "metrics.json",
            rollout_path=output / "rollout.json",
            output_dir=output,
        )

    def run_fixture(
        self,
        task_spec: TaskSpecInput,
        out_dir: str | Path,
        *,
        seed: int = 0,
        action_fn: Optional[ActionFn] = None,
        max_steps: Optional[int] = None,
        max_env_steps: int = 8,
        success_position: float = 3.0,
    ) -> RobotTaskRunResult:
        return self.run_local(
            task_spec,
            FixtureRobotEnv(max_steps=max_env_steps, success_position=success_position),
            out_dir,
            seed=seed,
            action_fn=action_fn,
            max_steps=max_steps,
        )

    def policy_artifact(
        self,
        task_spec: TaskSpecInput,
        *,
        artifact_id: str,
        created_by_run_id: str,
        checkpoint_uri: str | None = None,
        policy_format: str = "custom",
        eval_metrics: Mapping[str, Any] | None = None,
        rollout_artifacts: list[str] | None = None,
        backend_version: str | None = None,
        policy_kind: str = "rl_policy",
        inference_runtime: str | None = None,
        control_dt: float | None = None,
        latency_budget_ms: int | None = None,
        action_chunking: Mapping[str, Any] | None = None,
    ) -> PolicyArtifact:
        spec = self.validate(task_spec)
        payload: dict[str, Any] = {
            "schema_version": ROBOT_POLICY_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "task_spec_uri": f"task://{spec.task_id}",
            "task_spec_hash": spec.task_hash(),
            "checkpoint_uri": checkpoint_uri,
            "policy_format": policy_format,
            "observation_schema": spec.observation_space,
            "action_schema": spec.action_space,
            "robot_id": spec.robot_id,
            "simulator_backend": spec.simulator_backend,
            "backend_version": backend_version
            or str(spec.backend_config.get("image", spec.simulator_backend)),
            "eval_metrics": dict(eval_metrics or {}),
            "rollout_artifacts": list(rollout_artifacts or []),
            "created_by_run_id": created_by_run_id,
            "policy_kind": policy_kind,
            "inference_runtime": inference_runtime,
            "control_dt": control_dt,
            "latency_budget_ms": latency_budget_ms,
            "action_chunking": dict(action_chunking) if action_chunking is not None else None,
        }
        return PolicyArtifact.from_dict(
            {key: value for key, value in payload.items() if value is not None}
        )

    def write_policy_artifact(self, artifact: PolicyArtifact, path: str | Path) -> Path:
        return write_policy_artifact(path, artifact)
