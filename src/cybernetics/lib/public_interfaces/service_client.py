"""ServiceClient for Cybernetics API."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from cybernetics import types
from cybernetics.lib.client_connection_pool_type import ClientConnectionPoolType
from cybernetics.lib.credentials import resolve_api_key
from cybernetics.lib.public_interfaces.api_future import AwaitableConcurrentFuture
from cybernetics.lib.telemetry import Telemetry, capture_exceptions
from cybernetics.lib.telemetry_provider import TelemetryProvider

from ..api_future_impl import _APIFuture
from ..internal_client_holder import InternalClientHolder
from ..queue_state_logger import QueueStateLogger
from ..retry_handler import RetryConfig
from ..sync_only import sync_only

if TYPE_CHECKING:
    from .rest_client import RestClient
    from .sampling_client import SamplingClient
    from .training_client import TrainingClient

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)
OPTIMIZER_RESTORE_UNSUPPORTED_ERROR = (
    "This backend does not support optimizer-state restore. "
    "Use create_training_client_from_state() for weights-only restore instead."
)


class ServiceClient(TelemetryProvider):
    """The ServiceClient is the main entry point for the Cybernetics API. It provides methods to:
    - Query server capabilities and health status
    - Generate TrainingClient instances for model training workflows
    - Generate SamplingClient instances for text generation and inference
    - Generate RestClient instances for REST API operations like listing weights

    Args:
        user_metadata: Optional metadata attached to the created session.
        project_id: Optional project ID to attach to the created session.
        **kwargs: advanced options passed to the underlying HTTP client,
                 including API keys, headers, and connection settings.

    Example:
    ```python
    # Near instant
    client = ServiceClient()

    # Takes a moment as we initialize the model and assign resources
    training_client = client.create_lora_training_client(base_model="Qwen/Qwen3-8B")

    # Near-instant
    sampling_client = client.create_sampling_client(base_model="Qwen/Qwen3-8B")

    # Near-instant
    rest_client = client.create_rest_client()
    ```
    """

    def __init__(
        self,
        user_metadata: dict[str, str] | None = None,
        project_id: str | None = None,
        sample_dispatch_profile: str = "auto",
        **kwargs: Any,
    ):
        default_headers = _get_default_headers() | kwargs.pop("default_headers", {})
        self.holder = InternalClientHolder(
            user_metadata=user_metadata,
            project_id=project_id,
            sample_dispatch_profile=sample_dispatch_profile,
            **kwargs,
            default_headers=default_headers,
            _strict_response_validation=True,
        )
        logger.info(f"ServiceClient initialized for session {self.holder._session_id}")

    @property
    def session_id(self) -> str:
        """Worldlines session ID backing this client."""

        return self.holder.get_session_id()

    def _get_server_capabilities_submit(
        self,
    ) -> AwaitableConcurrentFuture[types.GetServerCapabilitiesResponse]:
        return self.holder.get_server_capabilities()

    def _get_scheduler_state_submit(
        self,
    ) -> AwaitableConcurrentFuture[types.GetSchedulerStateResponse]:
        return self.holder.get_scheduler_state()

    @sync_only
    @capture_exceptions(fatal=True)
    def get_server_capabilities(self) -> types.GetServerCapabilitiesResponse:
        """Query the server's supported features and capabilities.

        Returns:
        - `GetServerCapabilitiesResponse` with available models, features, and limits

        Example:
        ```python
        capabilities = service_client.get_server_capabilities()
        print(f"Supported models: {capabilities.supported_models}")
        print(f"Max batch size: {capabilities.max_batch_size}")
        ```
        """
        return self._get_server_capabilities_submit().result()

    @capture_exceptions(fatal=True)
    async def get_server_capabilities_async(self) -> types.GetServerCapabilitiesResponse:
        """Async version of get_server_capabilities."""
        return await self._get_server_capabilities_submit()

    @sync_only
    @capture_exceptions(fatal=True)
    def get_scheduler_state(self) -> types.GetSchedulerStateResponse:
        """Query the backend's current scheduler state and queue depths."""
        return self._get_scheduler_state_submit().result()

    @capture_exceptions(fatal=True)
    async def get_scheduler_state_async(self) -> types.GetSchedulerStateResponse:
        """Async version of get_scheduler_state."""
        return await self._get_scheduler_state_submit()

    def _create_lora_training_client_submit(
        self,
        base_model: str,
        rank: int,
        seed: int | None,
        train_mlp: bool,
        train_attn: bool,
        train_unembed: bool,
        user_metadata: dict[str, str] | None,
    ) -> AwaitableConcurrentFuture[TrainingClient]:
        assert any([train_mlp, train_attn, train_unembed]), (
            "At least one of train_mlp, train_attn, or train_unembed must be True"
        )
        session_id = self.holder.get_session_id()
        model_seq_id = self.holder.get_training_client_id()
        lora_config = types.LoraConfig(
            rank=rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
        )

        async def _create_lora_training_client_async():
            start_time = time.time()
            with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                request = types.CreateModelRequest(
                    session_id=session_id,
                    model_seq_id=model_seq_id,
                    base_model=base_model,
                    lora_config=lora_config,
                    user_metadata=user_metadata,
                )
                future = await client.models.create(request=request)
            create_model_response = await _APIFuture(
                types.CreateModelResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="CreateModel",
                queue_state_observer=QueueStateLogger(base_model, "Model creation"),
            ).result_async()
            model_id = create_model_response.model_id
            from .training_client import TrainingClient

            training_client = TrainingClient(
                self.holder, model_seq_id=model_seq_id, model_id=model_id
            )
            logger.info(f"TrainingClient initialized for model {model_id}")
            return training_client

        return self.holder.run_coroutine_threadsafe(_create_lora_training_client_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> TrainingClient:
        """Create a TrainingClient for LoRA fine-tuning.

        Args:
        - `base_model`: Name of the base model to fine-tune (e.g., "Qwen/Qwen3-8B")
        - `rank`: LoRA rank controlling the size of adaptation matrices (default 32)
        - `seed`: Random seed for initialization. None means random seed.
        - `train_mlp`: Whether to train MLP layers (default True)
        - `train_attn`: Whether to train attention layers (default True)
        - `train_unembed`: Whether to train unembedding layers (default True)
        - `user_metadata`: Optional metadata to attach to the training run
        - `timeout`: Optional seconds to wait for model creation/cold start.

        Returns:
        - `TrainingClient` configured for LoRA training

        Example:
        ```python
        training_client = service_client.create_lora_training_client(
            base_model="Qwen/Qwen3-8B",
            rank=16,
            train_mlp=True,
            train_attn=True
        )
        # Now use training_client.forward_backward() to train
        ```
        """
        return self._create_lora_training_client_submit(
            base_model,
            rank,
            seed,
            train_mlp,
            train_attn,
            train_unembed,
            user_metadata,
        ).result(timeout)

    @capture_exceptions(fatal=True)
    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ) -> TrainingClient:
        """Async version of create_lora_training_client."""
        return await self._create_lora_training_client_submit(
            base_model,
            rank,
            seed,
            train_mlp,
            train_attn,
            train_unembed,
            user_metadata,
        ).result_async()

    @sync_only
    @capture_exceptions(fatal=True)
    def create_training_client_from_state(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> TrainingClient:
        """Create a TrainingClient from saved model weights.

        This loads only the model weights, not optimizer state. To also restore
        optimizer state (e.g., Adam momentum), use create_training_client_from_state_with_optimizer.

        Args:
        - `path`: Cybernetics path to saved weights (e.g., "worldlines://run-id/weights/checkpoint-001")
        - `user_metadata`: Optional metadata to attach to the new training run

        Returns:
        - `TrainingClient` loaded with the specified weights

        Example:
        ```python
        # Resume training from a checkpoint (weights only, optimizer resets)
        training_client = service_client.create_training_client_from_state(
            "worldlines://run-id/weights/checkpoint-001"
        )
        # Continue training from the loaded state
        ```
        """
        rest_client = self.create_rest_client()
        # Use weights info endpoint which allows access to models with public checkpoints
        weights_info = rest_client.get_weights_info_by_worldlines_path(path).result()

        training_client = self.create_lora_training_client(
            base_model=weights_info.base_model,
            rank=weights_info.lora_rank,
            train_unembed=weights_info.train_unembed
            if weights_info.train_unembed is not None
            else True,
            train_mlp=weights_info.train_mlp if weights_info.train_mlp is not None else True,
            train_attn=weights_info.train_attn if weights_info.train_attn is not None else True,
            user_metadata=user_metadata,
        )

        training_client.load_state(path).result()
        return training_client

    @capture_exceptions(fatal=True)
    async def create_training_client_from_state_async(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> TrainingClient:
        """Async version of create_training_client_from_state."""
        rest_client = self.create_rest_client()
        # Use weights info endpoint which allows access to models with public checkpoints
        weights_info = await rest_client.get_weights_info_by_worldlines_path_async(path)

        # Right now all training runs are LoRa runs.
        assert weights_info.is_lora and weights_info.lora_rank is not None

        training_client = await self.create_lora_training_client_async(
            base_model=weights_info.base_model,
            rank=weights_info.lora_rank,
            train_unembed=weights_info.train_unembed
            if weights_info.train_unembed is not None
            else True,
            train_mlp=weights_info.train_mlp if weights_info.train_mlp is not None else True,
            train_attn=weights_info.train_attn if weights_info.train_attn is not None else True,
            user_metadata=user_metadata,
        )

        load_future = await training_client.load_state_async(path)
        await load_future.result_async()
        return training_client

    @sync_only
    @capture_exceptions(fatal=True)
    def create_training_client_from_state_with_optimizer(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> TrainingClient:
        """Create a TrainingClient from saved model weights and optimizer state.

        This is similar to create_training_client_from_state but also restores
        optimizer state (e.g., Adam momentum), which is useful for resuming
        training exactly where it left off.

        Args:
        - `path`: Cybernetics path to saved weights (e.g., "worldlines://run-id/weights/checkpoint-001")
        - `user_metadata`: Optional metadata to attach to the new training run

        Returns:
        - `TrainingClient` loaded with the specified weights and optimizer state

        Example:
        ```python
        # Resume training from a checkpoint with optimizer state
        training_client = service_client.create_training_client_from_state_with_optimizer(
            "worldlines://run-id/weights/checkpoint-001"
        )
        # Continue training with restored optimizer momentum
        ```
        """
        if self.holder.server_supports_optimizer_restore() is False:
            raise ValueError(OPTIMIZER_RESTORE_UNSUPPORTED_ERROR)
        rest_client = self.create_rest_client()
        # Use weights info endpoint which allows access to models with public checkpoints
        weights_info = rest_client.get_weights_info_by_worldlines_path(path).result()
        if weights_info.has_optimizer_state is False:
            raise ValueError(f"checkpoint at {path} does not contain optimizer state")

        training_client = self.create_lora_training_client(
            base_model=weights_info.base_model,
            rank=weights_info.lora_rank,
            train_unembed=weights_info.train_unembed
            if weights_info.train_unembed is not None
            else True,
            train_mlp=weights_info.train_mlp if weights_info.train_mlp is not None else True,
            train_attn=weights_info.train_attn if weights_info.train_attn is not None else True,
            user_metadata=user_metadata,
        )

        training_client.load_state_with_optimizer(path).result()
        return training_client

    @capture_exceptions(fatal=True)
    async def create_training_client_from_state_with_optimizer_async(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> TrainingClient:
        """Async version of create_training_client_from_state_with_optimizer."""
        if await self.holder.server_supports_optimizer_restore_async() is False:
            raise ValueError(OPTIMIZER_RESTORE_UNSUPPORTED_ERROR)
        rest_client = self.create_rest_client()
        # Use weights info endpoint which allows access to models with public checkpoints
        weights_info = await rest_client.get_weights_info_by_worldlines_path_async(path)
        if weights_info.has_optimizer_state is False:
            raise ValueError(f"checkpoint at {path} does not contain optimizer state")

        # Right now all training runs are LoRa runs.
        assert weights_info.is_lora and weights_info.lora_rank is not None

        training_client = await self.create_lora_training_client_async(
            base_model=weights_info.base_model,
            rank=weights_info.lora_rank,
            train_unembed=weights_info.train_unembed
            if weights_info.train_unembed is not None
            else True,
            train_mlp=weights_info.train_mlp if weights_info.train_mlp is not None else True,
            train_attn=weights_info.train_attn if weights_info.train_attn is not None else True,
            user_metadata=user_metadata,
        )

        load_future = await training_client.load_state_with_optimizer_async(path)
        await load_future.result_async()
        return training_client

    @capture_exceptions(fatal=True)
    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
        timeout: float | None = None,
    ) -> SamplingClient:
        """Create a SamplingClient for token or continuous-policy inference.

        Args:
        - `model_path`: Path to saved model weights (e.g., "worldlines://run-id/weights/checkpoint-001")
        - `base_model`: Name of base model to use (e.g., "Qwen/Qwen3-8B" or
          "pi0-droid"). If `model_path`
          is also provided, the checkpoint path takes precedence and the backend infers the
          base model from that checkpoint.
        - `retry_config`: Optional configuration for retrying failed requests
        - `timeout`: Optional seconds to wait for sampler creation/cold start.

        Returns:
        - `SamplingClient` configured for the selected model

        Raises:
            ValueError: If neither model_path nor base_model is provided

        Example:
        ```python
        # Use a base model
        sampling_client = service_client.create_sampling_client(
            base_model="Qwen/Qwen3-8B"
        )

        # Or use saved weights
        sampling_client = service_client.create_sampling_client(
            model_path="worldlines://run-id/weights/checkpoint-001"
        )
        ```
        """
        from .sampling_client import SamplingClient

        if model_path is None and base_model is None:
            raise ValueError("Either model_path or base_model must be provided")
        return SamplingClient.create(
            self.holder,
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
        ).result(timeout)

    @capture_exceptions(fatal=True)
    async def create_sampling_client_async(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> SamplingClient:
        """Async version of create_sampling_client."""
        from .sampling_client import SamplingClient

        if model_path is None and base_model is None:
            raise ValueError("Either model_path or base_model must be provided")
        return await SamplingClient.create(
            self.holder,
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
        )

    @capture_exceptions(fatal=True)
    def create_rest_client(self) -> RestClient:
        """Create a RestClient for REST API operations.

        The RestClient provides access to various REST endpoints for querying
        model information, checkpoints, sessions, and managing checkpoint visibility.

        Returns:
        - `RestClient` for accessing REST API endpoints

        Example:
        ```python
        rest_client = service_client.create_rest_client()

        # List checkpoints for a training run
        checkpoints = rest_client.list_checkpoints("run-id").result()

        # Get training run info
        training_run = rest_client.get_training_run("run-id").result()

        # Publish a checkpoint
        rest_client.publish_checkpoint_from_worldlines_path(
            "worldlines://run-id/weights/checkpoint-001"
        ).result()
        ```
        """
        from .rest_client import RestClient

        return RestClient(self.holder)

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()


def _get_default_headers() -> dict[str, str]:
    headers: dict[str, str] = {}

    # Authenticate to the control plane with the customer's cp_live_ key as a
    # Bearer token (resolved explicit -> CYBERNETICS_API_KEY -> CP_API_KEY ->
    # deprecated WORLDLINES_API_KEY -> stored login). Not X-API-Key: the control-plane CORS
    # allowlist is Authorization + Content-Type only.
    if (api_key := resolve_api_key()) and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {api_key}"

    if (
        client_id := os.environ.get("CLOUDFLARE_ACCESS_CLIENT_ID")
    ) and "CF-Access-Client-Id" not in headers:
        headers["CF-Access-Client-Id"] = client_id
    if (
        client_secret := os.environ.get("CLOUDFLARE_ACCESS_CLIENT_SECRET")
    ) and "CF-Access-Client-Secret" not in headers:
        headers["CF-Access-Client-Secret"] = client_secret
    return headers
