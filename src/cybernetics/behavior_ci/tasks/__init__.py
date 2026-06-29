"""Behavior CI Task Packs -- the platform-owned trust domain.

A *task pack* bundles everything that JUDGES a policy: the eval thresholds + scenarios, the
held-out perturbation bank, the pure action planner + outcome measurement, the in-session
Isaac grader, and the saved-scene env_id. Packs ship INSIDE this pip-installed SDK, so the
trust anchor sits outside the candidate policy repo: a policy PR can change only its opaque
checkpoint, never the judge.

``load_task(task_id)`` returns a :class:`Task` exposing the eval, the (pure) plan/measure
functions, the scenario lists, the grader source (read as TEXT only -- it imports omni.* and
must never be imported in-process), and the integrity lock. ``verify_candidate_copies``
sha256-compares any in-repo readability copies against the lock and raises on divergence, so
tampering with a candidate copy is both INERT (the pack bytes are authoritative) and LOUD.

Pure stdlib + pyyaml only; no new dependencies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib import import_module
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from ..schemas import ConfigError, ContractError

TASKS_PACKAGE = "cybernetics.behavior_ci.tasks"
TASKLOCK_SCHEMA_VERSION = "behavior-ci-tasklock/v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


@dataclass(frozen=True)
class TaskLock:
    """Integrity record for a task pack: sha256 of every judge-bearing file."""

    schema_version: str
    task_version: str
    grader_entrypoint: str
    digests: Dict[str, str]  # filename -> sha256 (pack-internal files)
    candidate_copies: Dict[str, str]  # repo-relative path -> expected sha256

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskLock":
        return cls(
            schema_version=data.get("schema_version", TASKLOCK_SCHEMA_VERSION),
            task_version=str(data.get("task_version", "0")),
            grader_entrypoint=data.get("grader_entrypoint", "behavior_ci_run_trial"),
            digests=dict(data.get("digests", {})),
            candidate_copies=dict(data.get("candidate_copies", {})),
        )


@dataclass(frozen=True)
class Task:
    """A loaded, content-addressed behavior-CI task pack."""

    task_id: str
    behavior: str
    robot: str
    world: str
    scene_env: str
    camera: str
    env_id: str
    grader_entrypoint: str
    action_contract: str
    checks: Dict[str, Any]
    visible: List[Dict[str, Any]]
    held_out: List[Dict[str, Any]]
    eval_dict: Dict[str, Any]
    grader_source: Optional[str]
    lock: Optional[TaskLock]
    _plan: Callable[[dict, dict], dict] = field(repr=False, default=None)
    _measure: Callable[[dict, dict], dict] = field(repr=False, default=None)
    _build_observation: Callable[[dict], dict] = field(repr=False, default=None)

    # -- the action/measurement contract (pure) --------------------------------------- #
    def build_observation(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        return self._build_observation(scenario)

    def plan(self, checkpoint: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        return self._plan(checkpoint, observation)

    def measure(self, trajectory: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
        return self._measure(trajectory, observation)


def _pack_resource(task_id: str):
    return files(f"{TASKS_PACKAGE}.{task_id}")


def _read_text(task_id: str, name: str) -> Optional[str]:
    res = _pack_resource(task_id) / name
    if not res.is_file():
        return None
    return res.read_text(encoding="utf-8")


def load_task(task_id: str) -> Task:
    """Load a packaged task by id. Raises ConfigError for an unknown/invalid pack."""
    if not task_id or "/" in task_id or "." in task_id:
        raise ConfigError(f"invalid task id {task_id!r}")
    try:
        pkg = _pack_resource(task_id)
    except ModuleNotFoundError:
        raise ConfigError(f"unknown behavior-ci task {task_id!r}")
    if not (pkg / "task.yaml").is_file():
        raise ConfigError(f"unknown behavior-ci task {task_id!r} (no task.yaml)")

    task_meta = yaml.safe_load(_read_text(task_id, "task.yaml"))
    eval_name = task_meta.get("eval", "eval.yaml")
    eval_dict = yaml.safe_load(_read_text(task_id, eval_name))

    # pure modules -- safe to import in-process
    mod = f"{TASKS_PACKAGE}.{task_id}"
    planner = import_module(f"{mod}.planner")
    measure = import_module(f"{mod}.measure")
    scenarios = import_module(f"{mod}.scenarios")

    # grader.py is read as TEXT only -- it imports omni.* and must NOT be imported here.
    grader_source = _read_text(task_id, task_meta.get("grader_module", "grader.py"))

    lock_text = _read_text(task_id, "lock.json")
    lock = TaskLock.from_dict(json.loads(lock_text)) if lock_text else None

    return Task(
        task_id=task_id,
        behavior=task_meta["behavior"],
        robot=task_meta["robot"],
        world=task_meta["world"],
        scene_env=task_meta["scene_env"],
        camera=task_meta["camera"],
        env_id=task_meta["env_id"],
        grader_entrypoint=task_meta["grader_entrypoint"],
        action_contract=task_meta.get("action_contract", "trajectory/v1"),
        checks=dict(eval_dict.get("checks", {})),
        visible=list(eval_dict.get("scenarios", [])),
        held_out=list(eval_dict.get("held_out", [])),
        eval_dict=eval_dict,
        grader_source=grader_source,
        lock=lock,
        _plan=planner.plan,
        _measure=measure.measure,
        _build_observation=scenarios.build_observation,
    )


def verify_candidate_copies(config_dir: Path, lock: TaskLock) -> None:
    """sha256-compare any in-repo readability copies against the pinned lock.

    A candidate copy that exists but diverges from the pinned digest raises ContractError
    (the pack bytes are authoritative for grading; this makes tampering LOUD too). A copy
    that is absent is fine -- the candidate may delete the readability copy entirely.
    """
    if lock is None:
        return
    config_dir = Path(config_dir)
    mismatches = []
    for relpath, expected in lock.candidate_copies.items():
        p = config_dir / relpath
        if not p.exists():
            continue
        actual = sha256_bytes(p.read_bytes())
        if actual != expected:
            mismatches.append(f"{relpath}: expected {expected[:12]}..., got {actual[:12]}...")
    if mismatches:
        raise ContractError(
            "candidate copy diverges from the pinned task lock (edit ignored for grading, "
            "and rejected here):\n  - " + "\n  - ".join(mismatches)
        )
