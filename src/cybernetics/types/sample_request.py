from typing import Optional

from pydantic import Field, model_validator
from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .droid_observation import DroidObservation
from .model_input import ModelInput
from .pi0_droid_dsrl_action import validate_pi0_initial_flow_noise
from .policy_conditioning import PolicyConditioning
from .sampling_params import SamplingParams
from .tensor_data import TensorData

__all__ = ["SampleRequest"]


class SampleRequest(StrictBase):
    num_samples: int = 1
    """Number of samples to generate"""

    prompt: ModelInput = Field(default_factory=ModelInput.empty)
    """Optional token prompt.

    Token LLM samplers use this as their full conditioning context. Continuous
    policies such as DreamZero may leave it empty and instead provide
    ``conditioning`` tensors.
    """

    conditioning: Optional[PolicyConditioning] = None
    """Runtime-native continuous-policy conditioning tensors.

    DreamZero expects RGB frames, proprioceptive state, masks, and embodiment
    conditioning at this boundary. Token-only samplers ignore this field.
    """

    droid_observation: Optional[DroidObservation] = None
    """Raw DROID observation for transforms owned by the selected policy backend."""

    pi0_initial_flow_noise: Optional[TensorData] = None
    """Advanced PI0-only ``[10, 32]`` initial flow-noise tensor.

    DSRL callers should use ``SamplingClient.sample_droid(dsrl_action=...)`` so
    the 32-dimensional controller action is expanded without ambiguity.
    """

    policy_mode: Optional[Literal["native", "sde"]] = None
    """Continuous-policy mode: native causal serving or recorded SDE rollout."""

    include_predicted_video: bool = False
    """Return one bounded predicted-video latent from the native joint pass."""

    sampling_params: SamplingParams

    base_model: Optional[str] = None
    """Optional base model name to sample from.

    Is inferred from model_path, if provided. If sampling against a base model, this
    is required.
    """

    model_path: Optional[str] = None
    """Optional worldlines:// path to your model weights or LoRA weights.

    If not provided, samples against the base model.
    """

    sampling_session_id: Optional[str] = None
    """Optional sampling session ID to use instead of model_path/base_model.

    If provided along with seq_id, the model configuration will be loaded from the
    sampling session. This is useful for multi-turn conversations.
    """

    seq_id: Optional[int] = None
    """Sequence ID within the sampling session.

    Required when sampling_session_id is provided. Used to generate deterministic
    request IDs for the sampling request.
    """

    prompt_logprobs: Optional[bool] = None
    """If set to `true`, computes and returns logprobs on the prompt tokens.

    Defaults to false.
    """

    topk_prompt_logprobs: int = 0
    """If set to a positive integer, returns the top-k logprobs for each prompt token."""

    completion_logprobs: Optional[bool] = None
    """E18: If true, the returned SampledSequence.logprobs is populated with the
    per-token logprob of each generated token. Off by default to save bandwidth."""

    type: Literal["sample"] = "sample"

    @model_validator(mode="after")
    def _validate_pi0_noise(self) -> "SampleRequest":
        if self.pi0_initial_flow_noise is not None:
            validate_pi0_initial_flow_noise(self.pi0_initial_flow_noise)
            if self.droid_observation is None:
                raise ValueError("pi0_initial_flow_noise requires droid_observation")
            if self.base_model not in (None, "pi0-droid"):
                raise ValueError("pi0_initial_flow_noise is supported only by pi0-droid")
        return self

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
