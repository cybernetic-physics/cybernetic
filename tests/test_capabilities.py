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


def test_pi0_dsrl_capability_fields_are_preserved() -> None:
    capabilities = GetServerCapabilitiesResponse(
        supported_models=[],
        pi0_initial_flow_noise_contract="pi0_initial_flow_noise_v1",
        pi0_initial_flow_noise_contract_version=1,
        pi0_initial_flow_noise_shape=[10, 32],
        pi0_initial_flow_noise_dtype="float32",
        pi0_dsrl_action_shape=[32],
        pi0_dsrl_action_dtype="float32",
        pi0_dsrl_expansion="repeat_across_action_horizon",
        base_policy_frozen=True,
    )

    assert capabilities.pi0_initial_flow_noise_contract_version == 1
    assert capabilities.pi0_initial_flow_noise_shape == [10, 32]
    assert capabilities.pi0_dsrl_action_shape == [32]
    assert capabilities.pi0_dsrl_expansion == "repeat_across_action_horizon"
    assert capabilities.base_policy_frozen is True
