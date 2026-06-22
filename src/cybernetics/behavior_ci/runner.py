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
        spec = EvalSpec.from_dict(_load_yaml(self._resolve_eval(eval_ref)))

        if manifest.behavior != spec.behavior:
            raise ConfigError(
                f"policy behavior '{manifest.behavior}' != eval behavior '{spec.behavior}'"
            )

        backend = select_backend(manifest)
        policy = backend.load(manifest)
        scene = SceneSpec(
            world=spec.world,
            scene_env=cfg.session.scene_env,
            camera=cfg.session.camera,
            robot=cfg.robot,
            env_id=cfg.session.env_id or os.environ.get(ENV_ID_ENV),
        )

        adapter = self._build_adapter(keep_session=keep_session)

        with adapter:
            adapter.prepare(scene)
            trials: List[TrialResult] = []
            for run in range(spec.runs):
                scenario = spec.scenarios[run] if run < len(spec.scenarios) else {}
                obs = adapter.run_trial(policy, run, scenario)
                trials.append(evaluate_trial(obs, spec))

            status, summary, metrics, failures, checks = aggregate(trials, spec)
            failed_run = next((t.run for t in trials if not t.passed), None)
            passed_run = next((t.run for t in trials if t.passed), None)
            replays = adapter.capture_replays(scene, failed_run, passed_run)
            replay_source = adapter.replay_source
            session_id = adapter.session_id

        self._enforce_provenance(replay_source)

        honesty = HonestyProvenance(
            simulator_adapter=adapter.adapter_id,
            replay_source=replay_source,
            policy_backend=manifest.backend,
            policy_backend_real_vla=manifest.real_vla,
            production_eval_path_used=False,
            scene_env=cfg.session.scene_env,
            camera=cfg.session.camera,
            session_id=session_id,
            notes=_provenance_note(adapter.adapter_id, replay_source),
        )

        commit = commit or _git_commit(self.config_dir)
        result = BehaviorCiResult(
            status=status,
            behavior=spec.behavior,
            robot=manifest.robot,
            world=spec.world,
            scene_env=cfg.session.scene_env,
            camera=cfg.session.camera,
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
        normalized = _normalized_manifest(manifest, spec, cfg, commit)
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

    def _build_adapter(self, keep_session: bool):
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
            return FixtureSimulatorAdapter(
                replay_dir=replay_dir, allow_placeholder_replays=allow_placeholder
            )

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
        return IsaacSessionAdapter(
            base_url=base_url,
            api_key=api_key,
            session=cfg.session,
            mcp_url=mcp_url,
            mcp_api_key=os.environ.get(MCP_API_KEY_ENV, api_key),
            workspace_id=cfg.workspace_id or os.environ.get(WORKSPACE_ENV),
            keep_session=keep_session,
        )

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
    manifest: PolicyManifest, spec: EvalSpec, cfg: BehaviorCiConfig, commit: str
) -> Dict[str, Any]:
    return {
        "schema_version": "behavior-ci-normalized/v1",
        "commit": commit,
        "project": cfg.project,
        "robot": cfg.robot,
        "policy": {
            "policy_id": manifest.policy_id,
            "display_filename": manifest.display_filename,
            "backend": manifest.backend,
            "controller": manifest.controller,
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
