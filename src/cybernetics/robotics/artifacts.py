"""Artifact writers for RobotTask SDK skeleton records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import PolicyArtifact, RobotRunRecord


def write_json_artifact(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a stable JSON artifact and return its path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def write_robot_run_record(path: str | Path, record: RobotRunRecord) -> Path:
    return write_json_artifact(path, record.to_dict())


def write_policy_artifact(path: str | Path, artifact: PolicyArtifact) -> Path:
    return write_json_artifact(path, artifact.to_dict())
