"""Behavior CI taskkit -- the dev-facing API for authoring tasks in your own repo.

    from cybernetics.behavior_ci.taskkit import Task, register_task, Check

Subclass :class:`Task`, decorate with ``@register_task("<id>")``, drop it in ``tasks/<id>/``.
No SDK release needed to define a new behavior task.
"""

from .base import Check, Metrics, Observation, Scenario, Task, Trajectory
from .loader import PackagedTaskLoader, RepoTaskLoader, TaskLoader
from .lock import TaskLock, compute_digests, sha256_bytes, sign_lock, verify_lock
from .registry import TASK_REGISTRY, get_task, register_task

__all__ = [
    "Task",
    "Check",
    "Scenario",
    "Observation",
    "Trajectory",
    "Metrics",
    "register_task",
    "get_task",
    "TASK_REGISTRY",
    "TaskLock",
    "compute_digests",
    "sign_lock",
    "verify_lock",
    "sha256_bytes",
    "TaskLoader",
    "PackagedTaskLoader",
    "RepoTaskLoader",
]
