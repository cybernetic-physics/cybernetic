"""Commands for validating and smoke-testing RobotTask specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from cybernetics.robotics import RobotContractError, RobotTasksClient

from ..context import CLIContext
from ..exceptions import CyberneticsCliError
from ..output import OutputBase


class RobotTaskOutput(OutputBase):
    def __init__(self, title: str, payload: dict[str, Any]) -> None:
        self.title = title
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    def get_title(self) -> str | None:
        return self.title

    def get_table_columns(self) -> list[str]:
        return ["Field", "Value"]

    def get_table_rows(self) -> list[list[str]]:
        return [[key, _format_value(value)] for key, value in self.payload.items()]


@click.group(name="robot-task")
def cli() -> None:
    """Validate and smoke-test RobotTask specs."""


@cli.command("validate")
@click.argument("task_spec", type=click.Path(exists=True, dir_okay=False))
@click.pass_obj
def validate_command(ctx: CLIContext | None, task_spec: str) -> None:
    """Validate a RobotTaskSpec JSON file."""
    client = RobotTasksClient()
    try:
        spec = client.load(task_spec)
    except (OSError, TypeError, json.JSONDecodeError, RobotContractError) as exc:
        raise CyberneticsCliError(str(exc)) from exc
    RobotTaskOutput(
        "RobotTask Spec",
        {
            "status": "valid",
            "task_id": spec.task_id,
            "robot_id": spec.robot_id,
            "simulator_backend": spec.simulator_backend,
            "task_spec_hash": spec.task_hash(),
        },
    ).print((ctx or CLIContext()).format)


@cli.command("run-fixture")
@click.argument("task_spec", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_dir", type=click.Path(file_okay=False))
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--max-steps", default=None, type=int, help="Episode step limit override.")
@click.option("--max-env-steps", default=8, show_default=True, type=int)
@click.option("--success-position", default=3.0, show_default=True, type=float)
@click.pass_obj
def run_fixture_command(
    ctx: CLIContext | None,
    task_spec: str,
    out_dir: str,
    seed: int,
    max_steps: int | None,
    max_env_steps: int,
    success_position: float,
) -> None:
    """Run the deterministic fixture RobotEnv and write run artifacts."""
    client = RobotTasksClient()
    try:
        spec = client.load(task_spec)
        result = client.run_fixture(
            spec,
            out_dir,
            seed=seed,
            max_steps=max_steps,
            max_env_steps=max_env_steps,
            success_position=success_position,
        )
    except (OSError, TypeError, json.JSONDecodeError, RobotContractError) as exc:
        raise CyberneticsCliError(str(exc)) from exc
    RobotTaskOutput("RobotTask Fixture Run", result.to_dict()).print(
        (ctx or CLIContext()).format
    )


@cli.command("policy-artifact")
@click.argument("task_spec", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=click.Path(dir_okay=False))
@click.option("--artifact-id", required=True, help="Policy artifact identifier.")
@click.option("--created-by-run-id", required=True, help="Source RobotRunRecord id.")
@click.option("--checkpoint-uri", default=None, help="Checkpoint URI, if available.")
@click.option("--policy-format", default="custom", show_default=True)
@click.option("--policy-kind", default="rl_policy", show_default=True)
@click.option("--backend-version", default=None)
@click.option("--rollout-artifact", multiple=True, help="Rollout artifact URI/path.")
@click.option(
    "--eval-metric",
    multiple=True,
    help="Metric as key=value. Numeric values are parsed when possible.",
)
@click.pass_obj
def policy_artifact_command(
    ctx: CLIContext | None,
    task_spec: str,
    out_path: str,
    artifact_id: str,
    created_by_run_id: str,
    checkpoint_uri: str | None,
    policy_format: str,
    policy_kind: str,
    backend_version: str | None,
    rollout_artifact: tuple[str, ...],
    eval_metric: tuple[str, ...],
) -> None:
    """Create a PolicyArtifact JSON file from a RobotTaskSpec."""
    client = RobotTasksClient()
    try:
        spec = client.load(task_spec)
        policy = client.policy_artifact(
            spec,
            artifact_id=artifact_id,
            created_by_run_id=created_by_run_id,
            checkpoint_uri=checkpoint_uri,
            policy_format=policy_format,
            policy_kind=policy_kind,
            backend_version=backend_version,
            rollout_artifacts=list(rollout_artifact),
            eval_metrics=_parse_metrics(eval_metric),
        )
        output_path = client.write_policy_artifact(policy, out_path)
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        RobotContractError,
    ) as exc:
        raise CyberneticsCliError(str(exc)) from exc
    RobotTaskOutput(
        "RobotTask Policy Artifact",
        {
            "artifact_id": policy.artifact_id,
            "task_spec_hash": policy.task_spec_hash,
            "policy_format": policy.policy_format,
            "policy_kind": policy.policy_kind,
            "output_path": str(output_path),
        },
    ).print((ctx or CLIContext()).format)


def _parse_metrics(values: tuple[str, ...]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if not sep or not key:
            raise ValueError("--eval-metric must be formatted as key=value")
        metrics[key] = _parse_scalar(raw)
    return metrics


def _parse_scalar(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)
