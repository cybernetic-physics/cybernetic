"""Typed contracts for Cybernetic Physics Behavior CI.

These are hand-written ``dataclasses`` (not pydantic models) on purpose: the
behavior-ci contract is a small, stable, JSON-serializable surface that must
round-trip identically across the ``pydantic>=1.9,<3`` range the SDK supports,
and customers validate the emitted bundle from non-Python CI too. Dataclasses +
explicit ``from_dict``/``to_dict`` keep that contract dependency-light and exact.

The artifact contract version is ``behavior-ci/v1``. Fields may be *added*
compatibly; existing keys consumed by reports/comments must not be renamed.
"""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

RESULT_SCHEMA_VERSION = "behavior-ci/v1"
METRICS_SCHEMA_VERSION = "behavior-ci-metrics/v1"
CONFIG_SCHEMA_VERSION = "cybernetic-behavior-ci-config/v1"
POLICY_SCHEMA_VERSION = "behavior-ci-policy/v2"
TASK_SCHEMA_VERSION = "behavior-ci-task/v1"
EVAL_SCHEMA_VERSION = "behavior-ci-eval/v1"

# Accepted policy-manifest schema versions. v2 is the closed-capability format (the policy
# carries only an opaque ``checkpoint`` + a pinned ``task``, and unknown top-level keys are
# rejected, so a policy cannot smuggle a grader-readable number or a ``session_entrypoint``
# capability). v1 (legacy ``controller`` manifests) is still parsed during migration.
POLICY_SCHEMA_VERSIONS = ("behavior-ci-policy/v1", "behavior-ci-policy/v2")

# Top-level keys a v2 policy manifest may carry. The schema is CLOSED: anything else is a
# ConfigError. This is a structural anti-gaming boundary -- there is no field through which
# a policy can hand the grader a trusted scalar or choose which function grades it.
POLICY_V2_ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "policy_id",
        "display_filename",
        "behavior",
        "robot",
        "backend",
        "task",
        "checkpoint",
        "provenance",
        "expected_demo_result",
        "notes",
    }
)

# Recognized policy backends. Only ``real-vla`` claims a real learned policy; the
# rest must report ``policy_backend_real_vla = false`` in provenance.
POLICY_BACKENDS = ("scripted-vla-shim", "gr00t-adapter-stub", "real-vla")

# Recognized simulator adapters.
SIMULATOR_ADAPTERS = ("fixture", "isaac-session")

# Recognized replay provenance values.
REPLAY_SOURCES = (
    "isaac-sim-session-video",
    "checked-in-demo-evidence",
    "fixture-generated",
    "none",
)

OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


class BehaviorCiError(Exception):
    """Base class for behavior-ci contract/validation errors."""


class ConfigError(BehaviorCiError):
    """A config/manifest/eval file is missing required fields or malformed."""


class ContractError(BehaviorCiError):
    """The produced bundle/provenance violates the behavior-ci contract."""


def _require(data: Dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ConfigError(f"{where}: missing required field '{key}'")
    return data[key]


def _check_schema(data: Dict[str, Any], expected: str, where: str) -> None:
    got = data.get("schema_version")
    if got != expected:
        raise ConfigError(f"{where}: schema_version must be '{expected}', got {got!r}")


# --------------------------------------------------------------------------- #
# Input contracts (repo-authored): config, policy manifest, task, eval
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Check:
    """One scalar pass/fail rule applied to a per-trial metric."""

    name: str
    metric: str
    operator: str
    value: Any
    required: bool = True

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "Check":
        op = _require(data, "operator", f"check '{name}'")
        if op not in OPS:
            raise ConfigError(f"check '{name}': unknown operator {op!r}; allowed {list(OPS)}")
        return cls(
            name=name,
            metric=_require(data, "metric", f"check '{name}'"),
            operator=op,
            value=_require(data, "value", f"check '{name}'"),
            required=bool(data.get("required", True)),
        )


@dataclass(frozen=True)
class EvalSpec:
    """The pinned behavior eval: world, trial count, domain variation, checks."""

    schema_version: str
    world: str
    behavior: str
    runs: int
    checks: Dict[str, Check]
    # Per-run obstacle placement / domain randomization, keyed by run index.
    # Keeps the fixture model honest-and-readable instead of hidden run-id hacks.
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    # Held-out perturbation bank (pinned task packs only): scenarios shipped inside the SDK
    # and never copied into a candidate eval, so a policy fit to the visible set but not
    # genuinely obstacle-relative fails here. Graded alongside ``scenarios``.
    held_out: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalSpec":
        _check_schema(data, EVAL_SCHEMA_VERSION, "eval spec")
        runs = int(_require(data, "runs", "eval spec"))
        if not 1 <= runs <= 64:
            raise ConfigError(f"eval spec: runs must be in [1, 64], got {runs}")
        checks_raw = _require(data, "checks", "eval spec")
        checks = {name: Check.from_dict(name, c) for name, c in checks_raw.items()}
        return cls(
            schema_version=data["schema_version"],
            world=_require(data, "world", "eval spec"),
            behavior=_require(data, "behavior", "eval spec"),
            runs=runs,
            checks=checks,
            scenarios=list(data.get("scenarios", [])),
            held_out=list(data.get("held_out", [])),
        )


@dataclass(frozen=True)
class PolicyManifest:
    """A policy reference (a ``.pt`` JSON manifest).

    v2 (closed capability): the policy carries only an opaque ``checkpoint`` consumed by the
    backend planner, plus a pinned ``task`` id. The grader NEVER reads a manifest field as
    ground truth, and unknown top-level keys are rejected -- so there is no scalar to
    self-attest with and no ``session_entrypoint`` to smuggle.

    v1 (legacy): a free-form ``controller`` dict. Still parsed during migration.
    """

    schema_version: str
    policy_id: str
    display_filename: str
    behavior: str
    robot: str
    backend: str
    task: Optional[str] = None
    checkpoint: Dict[str, Any] = field(default_factory=dict)
    controller: Dict[str, Any] = field(default_factory=dict)
    expected_demo_result: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyManifest":
        version = data.get("schema_version")
        if version not in POLICY_SCHEMA_VERSIONS:
            raise ConfigError(
                f"policy manifest: schema_version must be one of "
                f"{list(POLICY_SCHEMA_VERSIONS)}, got {version!r}"
            )
        backend = _require(data, "backend", "policy manifest")
        if backend not in POLICY_BACKENDS:
            raise ConfigError(
                f"policy manifest: backend must be one of {list(POLICY_BACKENDS)}, got {backend!r}"
            )
        common = dict(
            schema_version=version,
            policy_id=_require(data, "policy_id", "policy manifest"),
            display_filename=_require(data, "display_filename", "policy manifest"),
            behavior=_require(data, "behavior", "policy manifest"),
            robot=_require(data, "robot", "policy manifest"),
            backend=backend,
            expected_demo_result=data.get("expected_demo_result"),
            notes=data.get("notes") or data.get("training_note"),
        )
        if version == "behavior-ci-policy/v2":
            unknown = set(data) - POLICY_V2_ALLOWED_KEYS
            if unknown:
                raise ConfigError(
                    "policy manifest (v2): unknown top-level key(s) "
                    f"{sorted(unknown)}; the schema is closed -- a policy may carry only an "
                    "opaque 'checkpoint' and a pinned 'task' (no grader-readable params, no "
                    "session_entrypoint)."
                )
            return cls(
                task=_require(data, "task", "policy manifest"),
                checkpoint=dict(_require(data, "checkpoint", "policy manifest")),
                controller={},
                **common,
            )
        # v1 legacy
        return cls(
            task=None,
            checkpoint={},
            controller=dict(_require(data, "controller", "policy manifest")),
            **common,
        )

    @property
    def real_vla(self) -> bool:
        return self.backend == "real-vla"

    @property
    def is_v2(self) -> bool:
        return self.schema_version == "behavior-ci-policy/v2"

    @property
    def params(self) -> Dict[str, Any]:
        """Backend-facing opaque params: the v2 checkpoint, or the v1 controller."""
        return self.checkpoint if self.is_v2 else self.controller


@dataclass(frozen=True)
class TaskSpec:
    """The scene/task description; ties policy + eval to a saved environment."""

    schema_version: str
    task_id: str
    world: str
    scene_env: str
    robot: str
    behavior: str
    camera: str
    workspace: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskSpec":
        _check_schema(data, TASK_SCHEMA_VERSION, "task spec")
        return cls(
            schema_version=data["schema_version"],
            task_id=_require(data, "task_id", "task spec"),
            world=_require(data, "world", "task spec"),
            scene_env=_require(data, "scene_env", "task spec"),
            robot=_require(data, "robot", "task spec"),
            behavior=_require(data, "behavior", "task spec"),
            camera=_require(data, "camera", "task spec"),
            workspace=dict(data.get("workspace", {})),
        )


@dataclass(frozen=True)
class SessionConfig:
    """Hosted-session settings for the ``isaac-session`` adapter."""

    scene_env: str
    camera: str
    env_id: Optional[str] = None
    gpu_spec: Optional[str] = None
    # Optional dedicated-instance request (e.g. "vast"); only honored for callers
    # allowed to select runtime (system admins / service accounts). Defaults to
    # the platform warm pool when unset.
    runtime_provider: Optional[str] = None
    idle_timeout_minutes: int = 120
    ready_timeout_seconds: int = 900
    # Author-at-runtime: spawn this robot, upload this repo module, and call its
    # setup function to build the scene on a fresh blank session.
    spawn_robot: Optional[str] = None
    spawn_position: Optional[List[float]] = None
    module_path: Optional[str] = None
    setup_entrypoint: Optional[str] = None


@dataclass(frozen=True)
class BehaviorCiConfig:
    """Parsed ``cybernetic-behavior-ci.yaml`` — the central repo manifest."""

    schema_version: str
    project: str
    robot: str
    out: str
    require_replay_provenance: Optional[str]
    replay_source_dir: Optional[str]
    simulator_adapter: str
    base_url_env: str
    session: SessionConfig
    evals: Dict[str, str]
    # Non-secret connection config (checked into the repo, not GitHub secrets).
    # Only API keys are secrets; URLs/ids are public config.
    base_url: Optional[str] = None
    mcp_url: Optional[str] = None
    workspace_id: Optional[str] = None
    policy_backends: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehaviorCiConfig":
        _check_schema(data, CONFIG_SCHEMA_VERSION, "config")
        artifacts = dict(data.get("artifacts", {}))
        sim = dict(_require(data, "simulator", "config"))
        adapter = sim.get("adapter", "fixture")
        if adapter not in SIMULATOR_ADAPTERS:
            raise ConfigError(
                f"config: simulator.adapter must be one of {list(SIMULATOR_ADAPTERS)}, "
                f"got {adapter!r}"
            )
        sess = dict(sim.get("session", {}))
        require_prov = artifacts.get("require_replay_provenance")
        if require_prov is not None and require_prov not in REPLAY_SOURCES:
            raise ConfigError(
                f"config: artifacts.require_replay_provenance must be one of "
                f"{list(REPLAY_SOURCES)}, got {require_prov!r}"
            )
        evals = dict(_require(data, "evals", "config"))
        # Allow either {name: "path.yaml"} or {name: {path: "..."}}.
        eval_paths = {
            name: (spec if isinstance(spec, str) else _require(spec, "path", f"evals.{name}"))
            for name, spec in evals.items()
        }
        return cls(
            schema_version=data["schema_version"],
            project=_require(data, "project", "config"),
            robot=_require(data, "robot", "config"),
            out=artifacts.get("out", "artifacts/behavior-ci"),
            require_replay_provenance=require_prov,
            replay_source_dir=artifacts.get("replay_source_dir"),
            simulator_adapter=adapter,
            base_url_env=sim.get("base_url_env", "CYBERNETICS_BASE_URL"),
            base_url=sim.get("base_url"),
            mcp_url=sim.get("mcp_url"),
            workspace_id=sim.get("workspace_id"),
            session=SessionConfig(
                scene_env=sess.get("scene_env", ""),
                camera=sess.get("camera", ""),
                env_id=sess.get("env_id"),
                gpu_spec=sess.get("gpu_spec"),
                runtime_provider=sess.get("runtime_provider"),
                idle_timeout_minutes=int(sess.get("idle_timeout_minutes", 120)),
                ready_timeout_seconds=int(sess.get("ready_timeout_seconds", 900)),
                spawn_robot=sess.get("spawn_robot"),
                spawn_position=sess.get("spawn_position"),
                module_path=sess.get("module_path"),
                setup_entrypoint=sess.get("setup_entrypoint"),
            ),
            evals=eval_paths,
            policy_backends=dict(data.get("policy_backends", {})),
        )


# --------------------------------------------------------------------------- #
# Runtime/output contracts: observations, results, provenance
# --------------------------------------------------------------------------- #


@dataclass
class Event:
    run: int
    time_seconds: float
    code: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrialObservation:
    """Raw per-trial output of a (backend, simulator) rollout."""

    run: int
    metrics: Dict[str, float]
    events: List[Event] = field(default_factory=list)
    trajectory_id: str = ""


@dataclass
class CheckResult:
    passed: bool
    metric: str
    actual: Any
    operator: str
    expected: Any
    required: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrialResult:
    run: int
    passed: bool
    checks: Dict[str, CheckResult]
    metrics: Dict[str, float]
    events: List[Event] = field(default_factory=list)
    trajectory_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": self.run,
            "passed": self.passed,
            "checks": {k: v.to_dict() for k, v in self.checks.items()},
            "metrics": self.metrics,
            "events": [e.to_dict() for e in self.events],
            "trajectory_id": self.trajectory_id,
        }


@dataclass
class HonestyProvenance:
    """Explicit, never-overclaimed provenance written into ``result.json``."""

    simulator_adapter: str
    replay_source: str
    policy_backend: str
    policy_backend_real_vla: bool
    production_eval_path_used: bool
    scene_env: str
    camera: str
    artifact_contract_version: str = RESULT_SCHEMA_VERSION
    session_id: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BehaviorCiResult:
    """Top-level ``behavior-ci/v1`` result blob."""

    status: str  # "passed" | "failed"
    behavior: str
    robot: str
    world: str
    scene_env: str
    camera: str
    policy: str
    policy_id: str
    policy_backend: str
    commit: str
    summary: Dict[str, Any]
    checks: Dict[str, bool]
    metrics: Dict[str, Any]
    failures: List[Dict[str, Any]]
    trials: List[TrialResult]
    honesty: HonestyProvenance
    artifacts: Dict[str, str] = field(default_factory=dict)
    schema_version: str = RESULT_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "behavior": self.behavior,
            "robot": self.robot,
            "world": self.world,
            "scene_env": self.scene_env,
            "camera": self.camera,
            "policy": self.policy,
            "policy_id": self.policy_id,
            "policy_backend": self.policy_backend,
            "commit": self.commit,
            "summary": self.summary,
            "checks": self.checks,
            "metrics": self.metrics,
            "failures": self.failures,
            "artifacts": self.artifacts,
            "honesty": self.honesty.to_dict(),
        }
