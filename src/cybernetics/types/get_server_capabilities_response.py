from typing import List, Optional

from .._models import BaseModel

__all__ = ["GetServerCapabilitiesResponse", "SupportedModel"]


class SupportedModel(BaseModel):
    """Information about a model supported by the server."""

    model_name: Optional[str] = None
    """The name of the supported model."""


class GetServerCapabilitiesResponse(BaseModel):
    """Response containing the server's supported models and capabilities."""

    supported_models: List[SupportedModel]
    """List of models available on the server."""

    supports_sampling: Optional[bool] = None
    """Whether the backend supports sampling operations."""

    supports_training: Optional[bool] = None
    """Whether the backend supports training operations."""

    supports_optimizer_restore: Optional[bool] = None
    """Whether the backend supports restoring optimizer state from checkpoints."""

    supports_dynamic_lora_loading: Optional[bool] = None
    """Whether the backend supports dynamically loading LoRA adapters."""

    supports_custom_loss_v1: Optional[bool] = None
    """Whether the backend supports the official upstream-style custom-loss v1 flow."""

    supports_custom_loss_v2: Optional[bool] = None
    """Whether the backend supports the explicit backend-only custom-loss v2 flow."""

    runtime: Optional[str] = None
    """The backend runtime name, for example 'sglang' or 'vllm'."""

    scheduler_mode: Optional[str] = None
    """High-level description of the backend request scheduling mode."""

    scheduler_subject_key: Optional[str] = None
    """Identity dimension the backend currently uses for fairness or queue accounting."""

    recommended_max_outstanding_sampling_requests: Optional[int] = None
    """Suggested upper bound for outstanding sampling requests from one client."""

    recommended_max_outstanding_eval_requests: Optional[int] = None
    """Suggested upper bound for outstanding eval or batch-style requests from one client."""

    max_queued_requests_per_subject: Optional[int] = None
    """Suggested or enforced maximum queued requests per scheduler subject."""

    max_active_requests_per_subject: Optional[int] = None
    """Suggested or enforced maximum concurrently active requests per scheduler subject."""

    max_queued_interactive_sampling_requests: Optional[int] = None
    """Suggested or enforced maximum queued interactive-sampling requests across the backend queue."""

    max_queued_checkpoint_interactive_sampling_requests: Optional[int] = None
    """Suggested or enforced maximum queued checkpoint-backed interactive-sampling requests across the backend queue."""

    max_queued_bulk_sampling_requests: Optional[int] = None
    """Suggested or enforced maximum queued bulk-sampling requests across the backend queue."""

    max_queued_checkpoint_bulk_sampling_requests: Optional[int] = None
    """Suggested or enforced maximum queued checkpoint-backed bulk-sampling requests across the backend queue."""

    max_active_bulk_sampling_requests: Optional[int] = None
    """Suggested or enforced maximum concurrently active bulk-sampling requests across the backend queue."""

    max_queued_compute_logprobs_requests: Optional[int] = None
    """Suggested or enforced maximum queued compute-logprobs requests across the backend queue."""

    max_active_compute_logprobs_requests: Optional[int] = None
    """Suggested or enforced maximum concurrently active compute-logprobs requests across the backend queue."""

    sampling_overload_behavior: Optional[str] = None
    """How sampling overload is surfaced, for example queue-first or fail-fast."""

    compute_logprobs_overload_behavior: Optional[str] = None
    """How compute-logprobs overload is surfaced."""

    sampling_coalescing_window_ms: Optional[int] = None
    """Sampling microbatch coalescing window, in milliseconds, when supported."""

    compute_logprobs_coalescing_window_ms: Optional[int] = None
    """Compute-logprobs coalescing window, in milliseconds, when supported."""

    fairness_policy: Optional[str] = None
    """Name of the current fairness policy, if the backend advertises one."""

    supports_local_client_dispatch_profiles: Optional[bool] = None
    """Whether the backend advertises local-machine-aware client dispatch profiles."""

    loss_families: Optional[List[str]] = None
    """Loss families the active training runtime can execute."""

    dreamzero_rl_available: Optional[bool] = None
    """Whether the DreamZero runtime can execute RL losses on this backend image."""

    dreamzero_rl_unavailable_reason: Optional[str] = None
    """Actionable reason DreamZero RL losses are unavailable, when known."""

    pi0_initial_flow_noise_contract: Optional[str] = None
    """PI0 raw flow-noise wire contract, when the active runtime supports it."""

    pi0_initial_flow_noise_contract_version: Optional[int] = None
    """Version of the PI0 raw flow-noise wire contract."""

    pi0_initial_flow_noise_shape: Optional[List[int]] = None
    """Exact raw PI0 initial flow-noise tensor shape."""

    pi0_initial_flow_noise_dtype: Optional[str] = None
    """Exact raw PI0 initial flow-noise tensor dtype."""

    pi0_dsrl_action_shape: Optional[List[int]] = None
    """Controller-facing PI0 DSRL SAC action shape."""

    pi0_dsrl_action_dtype: Optional[str] = None
    """Controller-facing PI0 DSRL SAC action dtype."""

    pi0_dsrl_expansion: Optional[str] = None
    """How the DSRL action expands into PI0's raw flow-noise tensor."""

    base_policy_frozen: Optional[bool] = None
    """Whether the active continuous-policy runtime keeps base weights immutable."""
