"""Policy backends: how a policy reference becomes a controller the sim runs.

The product boundary is narrow on purpose. ``ScriptedPolicyBackend`` is the
deterministic demo/test backend (readable controller params, never a real
learned policy). A real VLA/GR00T adapter is an explicit future extension and is
selected only when a manifest declares ``backend: real-vla``.
"""

from .base import LoadedPolicy, PolicyBackend, select_backend
from .scripted import ScriptedPolicyBackend

__all__ = ["LoadedPolicy", "PolicyBackend", "ScriptedPolicyBackend", "select_backend"]
