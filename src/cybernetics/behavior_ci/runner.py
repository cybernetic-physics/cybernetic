"""Behavior CI orchestration: config -> adapter -> trials -> verdict -> bundle.

The runner owns the eval lifecycle but not simulator details (those live behind
:class:`SimulatorAdapter`) and not GitHub specifics (a thin Action/CLI owns the
PR comment). It is the single entrypoint behind ``cybernetics behavior-ci run``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..lib.credentials import resolve_api_key, resolve_base_url
from . import artifacts as artifacts_mod
from .backends.base import select_backend
from .evaluator import aggregate, evaluate_trial
from .schemas import (
    BehaviorCiConfig,
    BehaviorCiResult,
    ConfigError,
    ContractError,
    EvalSpec,
    HonestyProvenance,
    PolicyManifest,
    TrialResult,
)
from .simulators.base import SceneSpec
from .simulators.fixture import FixtureSimulatorAdapter
from .simulators.isaac_session import IsaacSessionAdapter
from .tasks import Task, load_task, verify_candidate_copies

MCP_URL_ENV = "CYBERNETICS_MCP_URL"
MCP_API_KEY_ENV = "CYBERNETICS_MCP_API_KEY"
WORKSPACE_ENV = "CYBERNETICS_WORKSPACE_ID"
ENV_ID_ENV = "BEHAVIOR_CI_ENV_ID"
# Hosted production control plane; the MCP gateway is served on the same host at
# /mcp. Used as the final fallback so the hosted adapter needs only an API key.
DEFAULT_BASE_URL = "https://api.cyberneticphysics.com"


class BehaviorCiRunner:
    """Runs one behavior eval against one policy and writes the artifact bundle."""

    def __init__(self, config: BehaviorCiConfig, config_dir: Path) -> None:
        self.config = config
        self.config_dir = config_dir

    @classmethod
    def from_config(cls, config_path: str | os.PathLike[str]) -> "BehaviorCiRunner":
        path = Path(config_path)
        config = BehaviorCiConfig.from_dict(_load_yaml(path))
        return cls(config=config, config_dir=path.resolve().parent)

    # -- public API -------------------------------------------------------- #

    def run_policy(
        self,
        policy_ref: str,
        eval_ref: str,
        out_dir: Optional[str | os.PathLike[str]] = None,
        commit: Optional[str] = None,
        keep_session: bool = False,
    ) -> BehaviorCiResult:
        cfg = self.config
        manifest = PolicyManifest.from_dict(_load_json(self._resolve(policy_ref)))

        # Pinned-task path (v2 manifest): the eval, scenario geometry, action contract,
        # grader and saved-scene env_id all come from the SDK-shipped Task Pack -- NOT from
        # any candidate-editable path. The policy may change only its opaque checkpoint, and
        # the environment measures the trajectory it emits. This is the trust boundary.
        task = load_task(manifest.task) if manifest.is_v2 else None

        if task is not None:
            if manifest.behavior != task.behavior:
                raise ConfigError(
                    f"policy behavior '{manifest.behavior}' != task behavior '{task.behavior}'"
                )
            self._enforce_pins(task)
            spec = EvalSpec.from_dict(task.eval_dict)
            scene = SceneSpec(
                world=task.world,
                scene_env=task.scene_env,
                camera=task.camera,
                robot=manifest.robot,
                env_id=task.env_id,
            )
            # Grade the published suite AND the held-out perturbation bank; all must pass.
            trial_scenarios = list(task.visible) + list(task.held_out)
        else:
            spec = EvalSpec.from_dict(_load_yaml(self._resolve_eval(eval_ref)))
            if manifest.behavior != spec.behavior:
                raise ConfigError(
                    f"policy behavior '{manifest.behavior}' != eval behavior '{spec.behavior}'"
                )
            scene = SceneSpec(
                world=spec.world,
                scene_env=cfg.session.scene_env,
                camera=cfg.session.camera,
                robot=cfg.robot,
                env_id=cfg.session.env_id or os.environ.get(ENV_ID_ENV),
            )
            trial_scenarios = [
                spec.scenarios[run] if run < len(spec.scenarios) else {} for run in range(spec.runs)
            ]

        backend = select_backend(manifest)
        policy = backend.load(manifest)
        adapter = self._build_adapter(keep_session=keep_session, task=task)

        with adapter:
            adapter.prepare(scene)
            trials: List[TrialResult] = []
            for run, scenario in enumerate(trial_scenarios):
                if task is not None:
                    observation = task.build_observation(scenario)
                    action = task.plan(policy.controller, observation)
                    obs = adapter.run_action_trial(action, run, observation, scenario)
                else:
                    obs = adapter.run_trial(policy, run, scenario)
                trials.append(evaluate_trial(obs, spec))

            status, summary, metrics, failures, checks = aggregate(trials, spec)
            failed_run = next((t.run for t in trials if not t.passed), None)
            passed_run = next((t.run for t in trials if t.passed), None)
            replays = adapter.capture_replays(scene, failed_run, passed_run)
            replay_source = adapter.replay_source
            session_id = adapter.session_id

        self._enforce_provenance(replay_source)

        scene_env = task.scene_env if task is not None else cfg.session.scene_env
        camera = task.camera if task is not None else cfg.session.camera

        honesty = HonestyProvenance(
            simulator_adapter=adapter.adapter_id,
            replay_source=replay_source,
            policy_backend=manifest.backend,
            policy_backend_real_vla=manifest.real_vla,
            # The pinned Task Pack path IS the production eval path (judge owned by the SDK,
            # not the candidate). The legacy candidate-resolved path is not.
            production_eval_path_used=task is not None,
            scene_env=scene_env,
            camera=camera,
            session_id=session_id,
            notes=_provenance_note(adapter.adapter_id, replay_source),
            task_id=manifest.task,
            task_version=(task.lock.task_version if task and task.lock else None),
            eval_sha256=(task.lock.digests.get("eval.yaml") if task and task.lock else None),
            grader_sha256=(task.lock.digests.get("grader.py") if task and task.lock else None),
            pins_verified=task is not None,
        )

        commit = commit or _git_commit(self.config_dir)
        result = BehaviorCiResult(
            status=status,
            behavior=spec.behavior,
            robot=manifest.robot,
            world=spec.world,
            scene_env=scene_env,
            camera=camera,
            policy=manifest.display_filename,
            policy_id=manifest.policy_id,
            policy_backend=manifest.backend,
            commit=commit,
            summary=summary,
            checks=checks,
            metrics=metrics,
            failures=failures,
            trials=trials,
            honesty=honesty,
        )

        out = Path(out_dir) if out_dir else (self.config_dir / cfg.out)
        normalized = _normalized_manifest(manifest, spec, cfg, commit, task)
        artifacts_mod.write_bundle(
            result,
            replays,
            spec,
            normalized,
            out,
            artifact_url=os.environ.get("BEHAVIOR_CI_ARTIFACT_URL", ""),
        )
        problems = artifacts_mod.validate_bundle(out, result)
        if problems:
            raise ContractError("artifact bundle invalid:\n  - " + "\n  - ".join(problems))
        return result

    # -- internals --------------------------------------------------------- #

    def _enforce_pins(self, task: Task) -> None:
        """For a pinned task, reject (and report) any in-repo readability copy that diverges
        from the SDK-shipped lock. The pack bytes are authoritative for grading regardless;
        this makes tampering with a candidate copy LOUD as well as inert."""
        if task.lock is not None:
            verify_candidate_copies(self.config_dir, task.lock)

    def _build_adapter(self, keep_session: bool, task: Optional[Task] = None):
        cfg = self.config
        if cfg.simulator_adapter == "fixture":
            if cfg.require_replay_provenance == "isaac-sim-session-video":
                raise ConfigError(
                    "config requires replay provenance 'isaac-sim-session-video' but the "
                    "simulator adapter is 'fixture'; switch the adapter to 'isaac-session' "
                    "or relax artifacts.require_replay_provenance."
                )
            replay_dir = self.config_dir / cfg.replay_source_dir if cfg.replay_source_dir else None
            allow_placeholder = cfg.require_replay_provenance != "checked-in-demo-evidence"
            adapter = FixtureSimulatorAdapter(
                replay_dir=replay_dir, allow_placeholder_replays=allow_placeholder
            )
            adapter.task = task
            return adapter

        # hosted isaac-session. Only API keys are secrets (env / login store);
        # URLs, workspace, and env id are non-secret config (config file first,
        # env var as fallback).
        api_key = resolve_api_key()
        if not api_key:
            raise ConfigError(
                "isaac-session adapter needs an API key; set CYBERNETICS_API_KEY "
                "(or CP_API_KEY) or run 'cybernetics auth login'."
            )
        # Defaults to hosted production; override via config (simulator.base_url)
        # or the CYBERNETICS_BASE_URL env var for a dev/staging stack.
        base_url = (
            cfg.base_url
            or resolve_base_url()
            or os.environ.get(cfg.base_url_env)
            or DEFAULT_BASE_URL
        )
        mcp_url = cfg.mcp_url or os.environ.get(MCP_URL_ENV) or base_url

        # Author-at-runtime: read the repo-provided session module so the adapter can
        # upload it and build the scene on a fresh blank session.
        module_source = None
        module_name = "behavior_ci_env"
        if task is not None:
            # Pinned task: the grader uploaded to Isaac is the PACK's bytes (verified by
            # digest), never a candidate-supplied module. env_id arrives via SceneSpec.
            if task.grader_source is None:
                raise ConfigError(f"task '{task.task_id}' ships no grader module")
            module_source = task.grader_source
        elif cfg.session.module_path:
            module_file = self._resolve(cfg.session.module_path)
            if not module_file.exists():
                raise ConfigError(f"simulator.session.module_path not found: {module_file}")
            module_source = module_file.read_text()
            module_name = module_file.stem

        adapter = IsaacSessionAdapter(
            base_url=base_url,
            api_key=api_key,
            session=cfg.session,
            mcp_url=mcp_url,
            mcp_api_key=os.environ.get(MCP_API_KEY_ENV, api_key),
            workspace_id=cfg.workspace_id or os.environ.get(WORKSPACE_ENV),
            keep_session=keep_session,
            spawn_robot=cfg.session.spawn_robot,
            spawn_position=cfg.session.spawn_position,
            module_source=module_source,
            module_name=module_name,
            setup_entrypoint=cfg.session.setup_entrypoint,
            runtime_provider=cfg.session.runtime_provider,
        )
        adapter.task = task
        return adapter

    def _enforce_provenance(self, replay_source: str) -> None:
        required = self.config.require_replay_provenance
        if required and replay_source != required:
            raise ContractError(
                f"replay provenance '{replay_source}' does not meet the required "
                f"'{required}' (artifacts.require_replay_provenance)."
            )

    def _resolve(self, ref: str) -> Path:
        p = Path(ref)
        return p if p.is_absolute() else self.config_dir / p

    def _resolve_eval(self, eval_ref: str) -> Path:
        # eval_ref may be a config-declared name or a direct path.
        if eval_ref in self.config.evals:
            return self._resolve(self.config.evals[eval_ref])
        return self._resolve(eval_ref)


def _normalized_manifest(
    manifest: PolicyManifest,
    spec: EvalSpec,
    cfg: BehaviorCiConfig,
    commit: str,
    task: Optional[Task] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "behavior-ci-normalized/v1",
        "commit": commit,
        "project": cfg.project,
        "robot": cfg.robot,
        "task": manifest.task,
        "policy": {
            "policy_id": manifest.policy_id,
            "display_filename": manifest.display_filename,
            "backend": manifest.backend,
            # v2 carries an opaque checkpoint (planner input); v1 carries a controller.
            "checkpoint" if manifest.is_v2 else "controller": manifest.params,
        },
        "eval": {
            "world": spec.world,
            "behavior": spec.behavior,
            "runs": spec.runs,
            "checks": {
                name: {
                    "metric": c.metric,
                    "operator": c.operator,
                    "value": c.value,
                    "required": c.required,
                }
                for name, c in spec.checks.items()
            },
            "scenarios": spec.scenarios,
        },
        "simulator": {
            "adapter": cfg.simulator_adapter,
            "scene_env": cfg.session.scene_env,
            "camera": cfg.session.camera,
            "env_id": cfg.session.env_id,
        },
    }


def _provenance_note(adapter_id: str, replay_source: str) -> str:
    if adapter_id == "isaac-session":
        return "Trials ran in a hosted Cybernetic Physics Isaac Sim session; replays captured from the named pass/fail camera."
    return (
        "Fixture mode: trial outcomes computed deterministically from readable controller "
        f"parameters (not a learned policy, not live Isaac). Replay source: {replay_source}."
    )


def _git_commit(cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ConfigError(
            "behavior-ci needs PyYAML to read YAML config/evals; "
            "install with: pip install 'cybernetic-physics[behavior-ci]'"
        ) from exc
    if not path.exists():
        raise ConfigError(f"file not found: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a YAML mapping")
    return data


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a JSON object")
    return data
