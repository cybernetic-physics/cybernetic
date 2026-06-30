"""Task registration + lookup. Task ids are OPAQUE (never a filesystem path)."""

from __future__ import annotations

import re
from typing import Dict, Type

from .base import Task

TASK_REGISTRY: Dict[str, Type[Task]] = {}
_ID_RE = re.compile(r"^[a-z0-9_]+$")


def register_task(task_id: str):
    """Class decorator: register a :class:`Task` subclass under ``task_id``.

    The id must match ``^[a-z0-9_]+$`` so it can never carry a path component (the loader
    resolves ``tasks/<id>/`` strictly by allowlist, not by interpolating untrusted input).
    """
    if not _ID_RE.match(task_id):
        raise ValueError(f"invalid task id {task_id!r}; must match ^[a-z0-9_]+$")

    def _decorate(cls: Type[Task]) -> Type[Task]:
        cls.task_id = task_id
        TASK_REGISTRY[task_id] = cls
        return cls

    return _decorate


def get_task(task_id: str) -> Task:
    """Instantiate the registered task for ``task_id`` (must already be imported)."""
    cls = TASK_REGISTRY.get(task_id)
    if cls is None:
        raise KeyError(f"task {task_id!r} is not registered")
    return cls()
