"""Simulator adapters: how trials are actually run and replays captured.

``FixtureSimulatorAdapter`` is the fast, deterministic, no-network path used for
local dev, examples, and the default public CI job. ``IsaacSessionAdapter`` is
the hosted path: it boots a Cybernetic Physics Isaac session from a saved
environment, drives it over the MCP bridge, and captures replay video from the
named pass/fail camera. Raw MCP JSON-RPC is hidden behind the adapter — public
users never write it by hand.
"""

from .base import ReplayResult, SceneSpec, SimulatorAdapter, looks_like_mp4
from .fixture import FixtureSimulatorAdapter
from .isaac_session import IsaacSessionAdapter, IsaacSessionError

__all__ = [
    "SceneSpec",
    "ReplayResult",
    "SimulatorAdapter",
    "FixtureSimulatorAdapter",
    "IsaacSessionAdapter",
    "IsaacSessionError",
    "looks_like_mp4",
]
