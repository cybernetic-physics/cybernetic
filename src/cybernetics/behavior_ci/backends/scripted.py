"""Deterministic scripted policy backend (demo/test only).

This is *not* a learned policy. It resolves a manifest's readable controller
parameters (e.g. ``clearance_margin_cm``, ``replan_enabled``) into a
:class:`LoadedPolicy`. The simulator adapter then turns those parameters into
trial outcomes. Behavior differences between policy versions live entirely in
these parameters — never in a hidden lookup by ``policy_id``.
"""

from __future__ import annotations

from ..schemas import PolicyManifest
from .base import LoadedPolicy


class ScriptedPolicyBackend:
    real_vla = False

    def __init__(self, backend_id: str = "scripted-vla-shim") -> None:
        self.backend_id = backend_id

    def load(self, manifest: PolicyManifest) -> LoadedPolicy:
        return LoadedPolicy(
            policy_id=manifest.policy_id,
            backend_id=self.backend_id,
            real_vla=False,
            controller=dict(manifest.controller),
        )
