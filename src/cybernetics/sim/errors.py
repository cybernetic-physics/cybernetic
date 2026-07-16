"""Simulation SDK exceptions."""


class SimulationError(RuntimeError):
    """A simulation operation failed."""


class SimulationLaunchError(SimulationError):
    """A created session failed before ``launch(wait=True)`` could return."""

    def __init__(
        self,
        session_id: str,
        *,
        stop_requested: bool,
        reason: str,
    ) -> None:
        self.session_id = session_id
        self.stop_requested = stop_requested
        cleanup = (
            "automatic stop requested"
            if stop_requested
            else "automatic stop failed; stop the session manually"
        )
        super().__init__(
            f"session {session_id} failed during launch readiness: {reason} ({cleanup})"
        )


class SimulationMCPError(SimulationError):
    """A session-scoped MCP operation failed."""
