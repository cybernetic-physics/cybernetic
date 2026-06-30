"""taskkit: in-repo task authoring API + integrity lock + RepoTaskLoader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cybernetics.behavior_ci.schemas import ContractError
from cybernetics.behavior_ci.taskkit import (
    Check,
    RepoTaskLoader,
    TaskLock,
    compute_digests,
    register_task,
    verify_lock,
)
from cybernetics.behavior_ci.taskkit.registry import TASK_REGISTRY

TASK_PY = """
from cybernetics.behavior_ci.taskkit import Task, Check, register_task

@register_task("demo_task")
class Demo(Task):
    behavior = "demo"
    robot = "R"
    world = "W"
    scene_env = "S"
    camera = "/Cam"
    env_id = "env_x"
    def scenarios(self):
        return ([{"k": 1}], [{"k": 2}])
    def build_observation(self, scenario):
        return {"k": scenario["k"], "time_budget_s": 30.0}
    def plan(self, checkpoint, observation):
        return {"v": checkpoint.get("v", 0)}
    def measure(self, trajectory, observation):
        return {"m": trajectory["v"] * observation["k"]}
    def checks(self):
        return {"ok": Check("m", "<=", 10)}
"""

TASK_TOML = """
schema_version = "behavior-ci-task/v2"
task_id = "demo_task"
task_version = "1"
module = "task.py"
lock = "task.lock"
"""


def _make_task(tmp_path: Path) -> Path:
    d = tmp_path / "tasks" / "demo_task"
    d.mkdir(parents=True)
    (d / "task.py").write_text(TASK_PY)
    (d / "task.toml").write_text(TASK_TOML)
    lock = TaskLock(
        task_id="demo_task",
        task_version="1",
        grader_entrypoint="behavior_ci_run_trial",
        digests=compute_digests(d, ["task.py", "task.toml"]),
    )
    (d / "task.lock").write_text(json.dumps(lock.to_dict()))
    return tmp_path / "tasks"


def test_register_task_id_validation():
    with pytest.raises(ValueError):
        register_task("Bad-Id!")(type("X", (), {}))


def test_repo_loader_projects_task(tmp_path: Path):
    TASK_REGISTRY.pop("demo_task", None)
    tasks_dir = _make_task(tmp_path)
    task = RepoTaskLoader(tasks_dir).load("demo_task")
    assert task.task_id == "demo_task" and task.behavior == "demo"
    assert task.env_id == "env_x" and task.camera == "/Cam"
    assert task.visible == [{"k": 1}] and task.held_out == [{"k": 2}]
    assert task.checks["ok"]["operator"] == "<="
    # the projected pure functions work
    obs = task.build_observation({"k": 3})
    assert task.measure(task.plan({"v": 2}, obs), obs) == {"m": 6}


def test_repo_loader_rejects_tamper(tmp_path: Path):
    TASK_REGISTRY.pop("demo_task", None)
    tasks_dir = _make_task(tmp_path)
    (tasks_dir / "demo_task" / "task.py").write_text(TASK_PY + "\n# tampered\n")
    with pytest.raises(ContractError):
        RepoTaskLoader(tasks_dir).load("demo_task")


def test_verify_lock_ok_and_mismatch(tmp_path: Path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "a.py").write_text("x = 1\n")
    lock = TaskLock(
        task_id="t", task_version="1", grader_entrypoint="e", digests=compute_digests(d, ["a.py"])
    )
    verify_lock(d, lock)  # ok
    (d / "a.py").write_text("x = 2\n")
    with pytest.raises(ContractError):
        verify_lock(d, lock)


def test_unknown_task_and_bad_id(tmp_path: Path):
    (tmp_path / "tasks").mkdir()
    from cybernetics.behavior_ci.schemas import ConfigError

    with pytest.raises(ConfigError):
        RepoTaskLoader(tmp_path / "tasks").load("nope")
    with pytest.raises(ConfigError):
        RepoTaskLoader(tmp_path / "tasks").load("../etc")
