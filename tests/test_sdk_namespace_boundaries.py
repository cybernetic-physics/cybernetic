from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SDK_ROOT / "src"


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(SRC_ROOT)
        if not env.get("PYTHONPATH")
        else str(SRC_ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=SDK_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_sim_imports_without_robotics_namespace() -> None:
    result = _run_import_probe(
        """
        import builtins
        import sys

        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cybernetics.robotics" or name.startswith("cybernetics.robotics."):
                raise AssertionError(f"cybernetics.sim imported robotics via {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        import cybernetics.sim as sim
        from cybernetics import Client

        assert sim.SimulationClient
        assert Client.__name__ == "Client"
        assert "cybernetics.robotics" not in sys.modules
        """
    )

    assert result.returncode == 0, result.stderr


def test_robotics_imports_without_sim_or_heavy_runtime_dependencies() -> None:
    result = _run_import_probe(
        """
        import builtins
        import sys

        forbidden = {
            "cybernetics.sim",
            "isaacsim",
            "locomujoco",
            "mujoco",
            "omni",
            "rclpy",
            "rospy",
            "worldlines",
            "cosmos",
        }
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level:
                return original_import(name, globals, locals, fromlist, level)
            if name in forbidden or any(name.startswith(item + ".") for item in forbidden):
                raise AssertionError(f"cybernetics.robotics imported {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        import cybernetics.robotics as robotics

        assert robotics.RobotTaskSpec
        assert "cybernetics.sim" not in sys.modules
        for module_name in forbidden - {"cybernetics.sim"}:
            assert module_name not in sys.modules
        """
    )

    assert result.returncode == 0, result.stderr


def test_composed_sdk_exposes_sim_and_robotics_without_namespace_conflict() -> None:
    result = _run_import_probe(
        """
        from cybernetics import Client
        from cybernetics.robotics import RobotTaskSpec
        from cybernetics.sim import SimulationClient

        client = Client(api_key="cp_live_test", base_url="https://api.test")
        assert isinstance(client.sim, SimulationClient)
        assert RobotTaskSpec.__name__ == "RobotTaskSpec"
        client.close()
        """
    )

    assert result.returncode == 0, result.stderr
