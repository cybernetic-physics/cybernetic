from __future__ import annotations

from cybernetics.types import GetServerCapabilitiesResponse


def test_dreamzero_rl_capability_fields_are_preserved() -> None:
    capabilities = GetServerCapabilitiesResponse(
        supported_models=[],
        loss_families=["cross_entropy"],
        dreamzero_rl_available=False,
        dreamzero_rl_unavailable_reason="DreamZero RL requires `groot.vla.rl`.",
    )

    assert capabilities.loss_families == ["cross_entropy"]
    assert capabilities.dreamzero_rl_available is False
    assert "groot.vla.rl" in capabilities.dreamzero_rl_unavailable_reason
