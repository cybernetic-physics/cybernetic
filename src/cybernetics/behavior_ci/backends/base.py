"""Policy backend protocol + selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

from ..schemas import ConfigError, PolicyManifest


@dataclass(frozen=True)
class LoadedPolicy:
    """A resolved, runnable policy handle passed to the simulator adapter."""

    policy_id: str
    backend_id: str
    real_vla: bool
    controller: Dict[str, Any] = field(default_factory=dict)

    def param(self, name: str, default: Any = None) -> Any:
        return self.controller.get(name, default)


@runtime_checkable
class PolicyBackend(Protocol):
    """Resolves a :class:`PolicyManifest` into a :class:`LoadedPolicy`.

    The backend owns *what the policy is*; the simulator adapter owns *running
    it*. Backends must report honest provenance via ``real_vla``.
    """

    backend_id: str
    real_vla: bool

    def load(self, manifest: PolicyManifest) -> LoadedPolicy: ...


def select_backend(manifest: PolicyManifest) -> PolicyBackend:
    """Pick the backend implied by ``manifest.backend``."""

    if manifest.real_vla:
        raise ConfigError(
            "policy backend 'real-vla' is not implemented yet. The first public "
            "demo ships the scripted controller; wire RealVlaBackend (CYB-68) to "
            "run a learned VLA/GR00T checkpoint."
        )
    # scripted-vla-shim and gr00t-adapter-stub both resolve to the scripted backend.
    from .scripted import ScriptedPolicyBackend

    return ScriptedPolicyBackend(backend_id=manifest.backend)
