"""``cybernetics behavior-ci`` — run a robot-behavior regression check.

Exit codes (consumed by the GitHub Action to turn the check red/green):

    0  behavior passed and artifacts are valid
    1  behavior regression (artifacts still written)
    2  invalid input/config
    3  hosted infrastructure/session failure
    4  artifact/report contract failure
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

EXIT_OK = 0
EXIT_BEHAVIOR_FAIL = 1
EXIT_INPUT = 2
EXIT_INFRA = 3
EXIT_CONTRACT = 4


@click.group()
def cli() -> None:
    """Run Cybernetic Physics Behavior CI on a robot policy."""


@cli.command("validate-config")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def validate_config(config_path: str) -> None:
    """Parse and validate a cybernetic-behavior-ci.yaml without running anything."""

    from cybernetics.behavior_ci.runner import BehaviorCiRunner
    from cybernetics.behavior_ci.schemas import BehaviorCiError

    try:
        runner = BehaviorCiRunner.from_config(config_path)
    except BehaviorCiError as exc:
        click.echo(f"invalid config: {exc}", err=True)
        sys.exit(EXIT_INPUT)
    cfg = runner.config
    click.echo(
        f"OK: project={cfg.project} robot={cfg.robot} adapter={cfg.simulator_adapter} "
        f"evals={list(cfg.evals)}"
    )


@cli.command("run")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--policy-ref", required=True, help="Path to a policy manifest (.pt).")
@click.option("--eval", "eval_ref", required=True, help="Eval name (from config) or path.")
@click.option("--out", "out_dir", default=None, help="Artifact bundle output dir.")
@click.option("--commit", default=None, help="Commit SHA to record (default: git HEAD).")
@click.option("--keep-session", is_flag=True, help="Do not stop the hosted session on exit.")
def run(
    config_path: str,
    policy_ref: str,
    eval_ref: str,
    out_dir: str | None,
    commit: str | None,
    keep_session: bool,
) -> None:
    """Run the eval and write the behavior-ci/v1 artifact bundle."""

    from cybernetics.behavior_ci.runner import BehaviorCiRunner
    from cybernetics.behavior_ci.schemas import ConfigError, ContractError
    from cybernetics.behavior_ci.simulators.isaac_session import IsaacSessionError

    try:
        runner = BehaviorCiRunner.from_config(config_path)
        result = runner.run_policy(
            policy_ref=policy_ref,
            eval_ref=eval_ref,
            out_dir=out_dir,
            commit=commit,
            keep_session=keep_session,
        )
    except ConfigError as exc:
        click.echo(f"config error: {exc}", err=True)
        sys.exit(EXIT_INPUT)
    except IsaacSessionError as exc:
        click.echo(f"infrastructure error: {exc}", err=True)
        sys.exit(EXIT_INFRA)
    except ContractError as exc:
        click.echo(f"artifact contract error: {exc}", err=True)
        sys.exit(EXIT_CONTRACT)

    s = result.summary
    verdict = "PASS" if result.passed else "FAIL"
    click.echo(
        f"[{verdict}] {result.policy}: {s['passed_runs']}/{s['total_runs']} trials passed "
        f"(adapter={result.honesty.simulator_adapter}, replay={result.honesty.replay_source})"
    )
    for f in result.failures:
        click.echo(f"  - run {f['run']} {f['code']}: {f['message']}")
    sys.exit(EXIT_OK if result.passed else EXIT_BEHAVIOR_FAIL)


@cli.command("render-comment")
@click.option("--artifact-dir", required=True, type=click.Path(exists=True))
def render_comment(artifact_dir: str) -> None:
    """Print the PR comment markdown from a produced artifact bundle."""

    comment = Path(artifact_dir) / "comment.md"
    if not comment.exists():
        click.echo(f"no comment.md in {artifact_dir}", err=True)
        sys.exit(EXIT_CONTRACT)
    click.echo(comment.read_text(), nl=False)
