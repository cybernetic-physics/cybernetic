"""Task loaders: resolve a task id to the runtime :class:`~cybernetics.behavior_ci.tasks.Task`.

- ``PackagedTaskLoader`` loads a task embedded in the SDK wheel (legacy path).
- ``RepoTaskLoader`` loads a task authored IN THE DOMAIN REPO under ``tasks/<id>/`` -- verifies
  its lock BEFORE importing any code, file-path-imports the pure ``task.py`` (running
  ``@register_task``), reads the hosted ``grader_isaac.py`` as TEXT (never imported here), and
  projects the author's :class:`taskkit.Task` instance into the frozen runtime Task. Defining a
  new task needs NO SDK release.
"""

from __future__ import annotations

import importlib.util
import tomllib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from ..schemas import ConfigError
from .lock import TaskLock, verify_lock
from .registry import _ID_RE, get_task


def _eval_dict_from_task(task, visible, held_out):
    checks = {}
    for name, c in task.checks().items():
        checks[name] = {
            "metric": c.metric,
            "operator": c.operator,
            "value": c.value,
            "required": c.required,
        }
    return {
        "schema_version": "behavior-ci-eval/v1",
        "world": task.world,
        "behavior": task.behavior,
        "runs": len(visible),
        "checks": checks,
        "scenarios": list(visible),
        "held_out": list(held_out),
    }


def _project(task, task_id, grader_source, lock):
    """Project a taskkit.Task instance into the frozen runtime Task dataclass."""
    from .. import tasks as tasks_pkg  # frozen Task dataclass lives here (avoid import cycle)

    visible, held_out = task.scenarios()
    eval_dict = _eval_dict_from_task(task, visible, held_out)
    return tasks_pkg.Task(
        task_id=task_id,
        behavior=task.behavior,
        robot=task.robot,
        world=task.world,
        scene_env=task.scene_env,
        camera=task.camera,
        env_id=task.env_id,
        grader_entrypoint=task.grader_entrypoint,
        action_contract=task.action_contract,
        checks=eval_dict["checks"],
        visible=list(visible),
        held_out=list(held_out),
        eval_dict=eval_dict,
        grader_source=grader_source,
        lock=lock,
        _plan=task.plan,
        _measure=task.measure,
        _build_observation=task.build_observation,
    )


class TaskLoader(ABC):
    @abstractmethod
    def load(self, task_id: str): ...


class PackagedTaskLoader(TaskLoader):
    """Legacy: a task embedded in the SDK package (cybernetics.behavior_ci.tasks.<id>)."""

    def load(self, task_id: str):
        from .. import tasks as tasks_pkg

        return tasks_pkg._load_packaged_task(task_id)


class RepoTaskLoader(TaskLoader):
    """A task authored in the domain repo under ``<tasks_dir>/<id>/``."""

    def __init__(
        self, tasks_dir: Path, pubkey_resolver: Optional[Callable[[str], Optional[bytes]]] = None
    ):
        self.tasks_dir = Path(tasks_dir)
        self.pubkey_resolver = pubkey_resolver

    def load(self, task_id: str):
        if not task_id or not _ID_RE.match(task_id):
            raise ConfigError(f"invalid task id {task_id!r}")
        task_dir = self.tasks_dir / task_id  # strict allowlist lookup; id can carry no path
        if not (task_dir / "task.toml").is_file():
            raise ConfigError(f"task {task_id!r} not found under {self.tasks_dir}")
        meta = tomllib.loads((task_dir / "task.toml").read_text())
        lock_name = meta.get("lock", "task.lock")
        lock_path = task_dir / lock_name
        if not lock_path.is_file():
            raise ConfigError(f"task {task_id!r} is missing its lock ({lock_name})")
        import json

        lock = TaskLock.from_dict(json.loads(lock_path.read_text()))
        # Verify integrity BEFORE importing any task code.
        pubkey = self.pubkey_resolver(task_id) if self.pubkey_resolver else None
        verify_lock(task_dir, lock, pubkey)

        module_name = meta.get("module", "task.py")
        grader_module = meta.get("grader_module", "grader_isaac.py")
        # File-path import the PURE tier; running @register_task registers the class.
        spec = importlib.util.spec_from_file_location(
            f"_behaviorci_task_{task_id}", task_dir / module_name
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        instance = get_task(task_id)
        # The hosted grader is read as TEXT only and never imported here (it imports omni.*).
        grader_path = task_dir / grader_module
        grader_source = grader_path.read_text() if grader_path.is_file() else None
        instance.grader_module = grader_module
        instance.grader_entrypoint = meta.get("grader_entrypoint", instance.grader_entrypoint)
        return _project(instance, task_id, grader_source, lock)
