"""Simulation SDK exceptions."""


class SimulationError(RuntimeError):
    """A simulation operation failed."""


class SimulationMCPError(SimulationError):
    """A session-scoped MCP operation failed."""
