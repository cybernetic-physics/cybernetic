"""TrainingClient for Cybernetics API."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Sequence, Tuple

from cybernetics import types
from cybernetics.lib.client_connection_pool_type import ClientConnectionPoolType
from cybernetics.lib.public_interfaces.api_future import APIFuture, AwaitableConcurrentFuture
from cybernetics.lib.telemetry import Telemetry, capture_exceptions
from cybernetics.lib.telemetry_provider import TelemetryProvider

from ..api_future_impl import (
    _APIFuture,
    _CombinedAPIFuture,
)
from ..chunked_fwdbwd_helpers import combine_fwd_bwd_output_results
from ..queue_state_logger import QueueStateLogger
from ..retry_handler import RetryConfig
from ..sync_only import sync_only
from .sampling_client import SamplingClient, _load_tokenizer_from_model_info

try:
    import torch
except ImportError:
    torch = None


if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ..internal_client_holder import InternalClientHolder

# pyright: reportPrivateImportUsage=false

logger = logging.getLogger(__name__)

# FwdBwdChunkSize
MAX_CHUNK_LEN = 1024
MAX_CHUNK_BYTES_COUNT = 5000000
MODEL_ID_NOT_SET_ERROR = "model_id must be set before calling forward. Try initializing the TrainingClient with a model_id by either calling create_lora_training_client on the ServiceClient, or initiliazing the TrainingClient with an existing model_id."
INFERENCE_ONLY_BACKEND_ERROR = (
    "This backend does not support Cybernetics training operations. "
    "Use create_sampling_client() or a checkpoint-backed sampling workflow instead."
)
OPTIMIZER_RESTORE_UNSUPPORTED_ERROR = (
    "This backend does not support optimizer-state restore. "
    "Use load_state() for weights-only restore instead."
)
CUSTOM_LOSS_V2_UNSUPPORTED_ERROR = (
    "This backend does not support backend-only custom loss v2. "
    "Check get_server_capabilities().supports_custom_loss_v2 before calling "
    "forward_backward_custom_v2()."
)

# Type alias for custom loss functions.
# Args: (data: List[Datum], model_outputs: List[Any]) -> (loss: Any, metrics: Dict[str, float])
CustomLossFnV1 = Callable[[List[types.Datum], List[Any]], Tuple[Any, Dict[str, float]]]
CustomLossFnV2 = Callable[[types.CustomLossContextV2], types.CustomLossOutputV2]

_SUPPORTED_CUSTOM_BACKEND_LOSS_FNS = frozenset({"cross_entropy"})
_CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE: dict[Literal["logprobs"], types.LossFnType] = {
    "logprobs": "cross_entropy",
}
_ALL_CUSTOM_LOSS_V2_INPUTS = frozenset(
    {
        "target_logprobs",
        "prompt_logprobs",
        "behavior_logprobs",
        "reference_logprobs",
        "candidate_logprobs",
        "values",
        "advantages",
        "returns",
    }
)
_SUPPORTED_CUSTOM_LOSS_V2_INPUTS = frozenset({"target_logprobs"})
_SUPPORTED_CUSTOM_LOSS_V2_LAYOUTS = frozenset({"padded"})


class TrainingClient(TelemetryProvider):
    """Client for training ML models with forward/backward passes and optimization.

    The TrainingClient corresponds to a fine-tuned model that you can train and sample from.
    You typically get one by calling `service_client.create_lora_training_client()`.
    Key methods:
    - forward_backward() - compute gradients for training
    - optim_step() - update model parameters with Adam optimizer
    - save_weights_and_get_sampling_client() - export trained model for inference

    Args:
    - `holder`: Internal client managing HTTP connections and async operations
    - `model_id`: Unique identifier for the model to train. Required for training operations.

    Example:
    ```python
    training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-8B")
    fwdbwd_future = training_client.forward_backward(training_data, "cross_entropy")
    optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
    fwdbwd_result = fwdbwd_future.result()  # Wait for gradients
    optim_result = optim_future.result()    # Wait for parameter update
    sampling_client = training_client.save_weights_and_get_sampling_client("my-model")
    ```
    """

    def __init__(self, holder: InternalClientHolder, model_seq_id: int, model_id: types.ModelID):
        self.holder = holder
        self.model_id = model_id

        self._training_client_id: int = model_seq_id

        self._request_id_lock: threading.Lock = threading.Lock()
        self._request_id_counter: int = 0

        self._turn_counter: int = 0
        self._turn_waiters: dict[int, asyncio.Event] = {}

        self._queue_state_logger = QueueStateLogger(str(model_id), "Training")

    # Reserves a request id for a request. Requests are to be executed in the order of request ids.
    def _get_request_id(self) -> int:
        with self._request_id_lock:
            request_id = self._request_id_counter
            self._request_id_counter += 1
            return request_id

    # Waits for the turn for a given request id to be executed.
    # This has to be used via a with statement so that the turn is released
    # only after current request was successfully dispatched.
    @asynccontextmanager
    async def _take_turn(self, request_id: int):
        assert self._turn_counter <= request_id, "Same request id cannot be taken twice"

        if self._turn_counter < request_id:
            try:
                event = asyncio.Event()
                self._turn_waiters[request_id] = event
                await event.wait()
            finally:
                del self._turn_waiters[request_id]

        assert self._turn_counter == request_id

        try:
            yield
        finally:
            self._turn_counter += 1
            if self._turn_counter in self._turn_waiters:
                self._turn_waiters[self._turn_counter].set()

    def _guaranteed_model_id(self) -> types.ModelID:
        assert self.model_id is not None, MODEL_ID_NOT_SET_ERROR
        return self.model_id

    def _assert_training_supported(self) -> None:
        if self.holder.server_supports_training() is False:
            raise ValueError(INFERENCE_ONLY_BACKEND_ERROR)

    async def _assert_training_supported_async(self) -> None:
        if await self.holder.server_supports_training_async() is False:
            raise ValueError(INFERENCE_ONLY_BACKEND_ERROR)

    def _assert_optimizer_restore_supported(self) -> None:
        if self.holder.server_supports_optimizer_restore() is False:
            raise ValueError(OPTIMIZER_RESTORE_UNSUPPORTED_ERROR)

    async def _assert_optimizer_restore_supported_async(self) -> None:
        if await self.holder.server_supports_optimizer_restore_async() is False:
            raise ValueError(OPTIMIZER_RESTORE_UNSUPPORTED_ERROR)

    def _estimate_bytes_count(self, datum: types.Datum) -> int:
        return self.holder.estimate_bytes_count_in_model_input(datum.model_input) + sum(
            len(value.data) * 10 for _, value in datum.loss_fn_inputs.items()
        )

    def _chunked_requests_generator(
        self, data: List[types.Datum]
    ) -> Generator[List[types.Datum], None, None]:
        current_chunk: List[types.Datum] = []
        current_chunk_bytes_count = 0

        for datum in data:
            estimated_bytes_count = self._estimate_bytes_count(datum)
            if (
                len(current_chunk) > 0
                and current_chunk_bytes_count + estimated_bytes_count > MAX_CHUNK_BYTES_COUNT
            ) or (len(current_chunk) == MAX_CHUNK_LEN):
                yield current_chunk
                current_chunk = []
                current_chunk_bytes_count = 0

            current_chunk.append(datum)
            current_chunk_bytes_count += estimated_bytes_count

        if len(current_chunk) > 0:
            yield current_chunk

    def _chunked_requests(self, data: List[types.Datum]) -> List[tuple[int, List[types.Datum]]]:
        return [(self._get_request_id(), chunk) for chunk in self._chunked_requests_generator(data)]

    async def _send_single_forward_request(
        self,
        request_id: int,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Dict[str, float] | None = None,
    ):
        request = types.ForwardRequest(
            forward_input=types.ForwardBackwardInput(
                data=data, loss_fn=loss_fn, loss_fn_config=loss_fn_config
            ),
            model_id=self._guaranteed_model_id(),
            seq_id=request_id + 1,
        )
        with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            return await client.training.forward(
                request=request,
            )

    @capture_exceptions(fatal=True)
    def forward(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Compute forward pass without gradients.

        Args:
        - `data`: List of training data samples
        - `loss_fn`: Loss function type (e.g., "cross_entropy")
        - `loss_fn_config`: Optional configuration for the loss function

        Returns:
        - `APIFuture` containing the forward pass outputs and loss

        Example:
        ```python
        data = [types.Datum(
            model_input=types.ModelInput.from_ints(tokenizer.encode("Hello")),
            loss_fn_inputs={"target_tokens": types.ModelInput.from_ints(tokenizer.encode("world"))}
        )]
        future = training_client.forward(data, "cross_entropy")
        result = await future
        print(f"Loss: {result.loss}")
        ```
        """
        self._assert_training_supported()
        requests = self._chunked_requests(data)

        @capture_exceptions(fatal=True)
        async def _forward_async():
            start_time = time.time()
            futures = []
            for request_id, data in requests:
                async with self._take_turn(request_id):
                    untyped_future = await self.holder.execute_with_retries(
                        self._send_single_forward_request, request_id, data, loss_fn, loss_fn_config
                    )
                api_future = _APIFuture(
                    types.ForwardBackwardOutput,
                    self.holder,
                    untyped_future,
                    request_start_time=start_time,
                    request_type="Forward",
                    queue_state_observer=self._queue_state_logger,
                )
                futures.append(api_future)
            return await _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)

        return self.holder.run_coroutine_threadsafe(_forward_async())

    async def forward_async(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Async version of forward."""
        await self._assert_training_supported_async()

        async def _forward_on_holder() -> APIFuture[types.ForwardBackwardOutput]:
            requests = self._chunked_requests(data)
            futures = []
            start_time = time.time()

            for request_id, chunk in requests:
                async with self._take_turn(request_id):
                    untyped_future = await self.holder.execute_with_retries(
                        self._send_single_forward_request,
                        request_id,
                        chunk,
                        loss_fn,
                        loss_fn_config,
                    )
                api_future = _APIFuture(
                    types.ForwardBackwardOutput,
                    self.holder,
                    untyped_future,
                    request_start_time=start_time,
                    request_type="Forward",
                    queue_state_observer=self._queue_state_logger,
                )
                futures.append(api_future)

            return _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)

        return await self.holder.run_coroutine_threadsafe(_forward_on_holder()).result_async()

    async def _send_single_forward_backward_request(
        self,
        request_id: int,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Dict[str, float] | None = None,
    ):
        request = types.ForwardBackwardRequest(
            forward_backward_input=types.ForwardBackwardInput(
                data=data, loss_fn=loss_fn, loss_fn_config=loss_fn_config
            ),
            model_id=self._guaranteed_model_id(),
            seq_id=request_id + 1,
        )
        with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            return await client.training.forward_backward(
                request=request,
            )

    @capture_exceptions(fatal=True)
    def forward_backward(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Compute forward pass and backward pass to calculate gradients.

        Args:
        - `data`: List of training data samples
        - `loss_fn`: Loss function type (e.g., "cross_entropy")
        - `loss_fn_config`: Optional configuration for the loss function

        Returns:
        - `APIFuture` containing the forward/backward outputs, loss, and gradients

        Example:
        ```python
        data = [types.Datum(
            model_input=types.ModelInput.from_ints(tokenizer.encode("Hello")),
            loss_fn_inputs={"target_tokens": types.ModelInput.from_ints(tokenizer.encode("world"))}
        )]

        # Compute gradients
        fwdbwd_future = training_client.forward_backward(data, "cross_entropy")

        # Update parameters
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=1e-4)
        )

        fwdbwd_result = await fwdbwd_future
        print(f"Loss: {fwdbwd_result.loss}")
        ```
        """
        self._assert_training_supported()
        requests = self._chunked_requests(data)

        @capture_exceptions(fatal=True)
        async def _forward_backward_async():
            futures = []
            start_time = time.time()

            for request_id, data in requests:
                async with self._take_turn(request_id):
                    untyped_future = await self.holder.execute_with_retries(
                        self._send_single_forward_backward_request,
                        request_id,
                        data,
                        loss_fn,
                        loss_fn_config,
                    )
                api_future = _APIFuture(
                    types.ForwardBackwardOutput,
                    self.holder,
                    untyped_future,
                    request_start_time=start_time,
                    request_type="ForwardBackward",
                    queue_state_observer=self._queue_state_logger,
                )
                futures.append(api_future)

            return await _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)

        return self.holder.run_coroutine_threadsafe(_forward_backward_async())

    async def forward_backward_async(
        self,
        data: List[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: Dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Async version of forward_backward."""
        await self._assert_training_supported_async()

        async def _forward_backward_on_holder() -> APIFuture[types.ForwardBackwardOutput]:
            requests = self._chunked_requests(data)
            futures = []
            start_time = time.time()

            for request_id, chunk in requests:
                async with self._take_turn(request_id):
                    untyped_future = await self.holder.execute_with_retries(
                        self._send_single_forward_backward_request,
                        request_id,
                        chunk,
                        loss_fn,
                        loss_fn_config,
                    )
                api_future = _APIFuture(
                    types.ForwardBackwardOutput,
                    self.holder,
                    untyped_future,
                    request_start_time=start_time,
                    request_type="ForwardBackward",
                    queue_state_observer=self._queue_state_logger,
                )
                futures.append(api_future)

            return _CombinedAPIFuture(futures, combine_fwd_bwd_output_results, self.holder)

        return await self.holder.run_coroutine_threadsafe(
            _forward_backward_on_holder()
        ).result_async()

    @sync_only
    @capture_exceptions(fatal=True)
    def forward_backward_custom(
        self,
        data: List[types.Datum],
        loss_fn: CustomLossFnV1,
        *,
        loss_type_input: Literal["logprobs"] = "logprobs",
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Compute forward/backward with a custom loss function.

        Allows you to define custom loss functions that operate on log probabilities.
        The custom function receives logprobs and computes loss and gradients.

        Args:
        - `data`: List of training data samples
        - `loss_fn`: Custom loss function that takes (data, logprobs) and returns (loss, metrics)
        - `loss_type_input`: Input space for `loss_fn`. Currently the only supported value is `"logprobs"`.

        Returns:
        - `APIFuture` containing the forward/backward outputs with custom loss

        Example:
        ```python
        def custom_loss(data, logprobs_list):
            # Custom loss computation
            loss = torch.mean(torch.stack([torch.mean(lp) for lp in logprobs_list]))
            metrics = {"custom_metric": loss.item()}
            return loss, metrics

        future = training_client.forward_backward_custom(data, custom_loss)
        result = future.result()
        print(f"Loss outputs: {result.loss_fn_outputs}")
        print(f"Metrics: {result.metrics}")
        ```
        """
        self._assert_training_supported()
        return self.holder.run_coroutine_threadsafe(
            self.forward_backward_custom_async(
                data,
                loss_fn,
                loss_type_input=loss_type_input,
            )
        ).result()

    @capture_exceptions(fatal=True)
    async def forward_backward_custom_async(
        self,
        data: List[types.Datum],
        loss_fn: CustomLossFnV1,
        *,
        loss_type_input: Literal["logprobs"] = "logprobs",
    ) -> APIFuture[types.ForwardBackwardOutput]:
        """Async version of forward_backward_custom."""
        await self._assert_training_supported_async()
        if torch is None:
            raise ImportError("PyTorch is not installed. Cannot run custom forward_backward.")

        if loss_type_input not in _CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE:
            supported = ", ".join(sorted(_CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE))
            raise ValueError(
                f"Unsupported loss_type_input={loss_type_input!r}. "
                f"Supported values are: {supported}"
            )

        surrogate_loss_fn = _CUSTOM_BACKEND_LOSS_FN_BY_INPUT_TYPE[loss_type_input]

        forward_data = self._get_custom_loss_forward_data(data, surrogate_loss_fn)

        # First do a forward pass and get logprobs
        forward_future = await self.holder.run_coroutine_threadsafe(
            self.forward_async(
                forward_data,
                surrogate_loss_fn,
                None,
            )
        ).result_async()
        forward_result = await forward_future.result_async()
        logprobs_list = []
        for out in forward_result.loss_fn_outputs:
            logprob = torch.tensor(out["logprobs"].data)
            if out["logprobs"].shape is not None:
                logprob = logprob.reshape(out["logprobs"].shape)
            logprob = logprob.clone().detach().requires_grad_(True)
            logprobs_list.append(logprob)

        # Now apply user-provided function
        loss, metrics = loss_fn(data, logprobs_list)
        loss.backward()
        grads = []
        for logprob in logprobs_list:
            if logprob.grad is None:
                raise ValueError("No gradient computed for logprob tensor")
            grads.append(logprob.grad)

        linear_loss_data = []
        for datum, grad in zip(data, grads, strict=True):
            loss_fn_inputs: Any = {
                "target_tokens": datum.loss_fn_inputs["target_tokens"],
                # Backend CE is L = sum(-logprobs * weights), so to backpropagate a
                # client-side custom loss C(logprobs) we must send weights = -dC/dlogprobs.
                "weights": -grad,
            }
            linear_loss_data.append(
                types.Datum(
                    model_input=datum.model_input,
                    loss_fn_inputs=loss_fn_inputs,
                )
            )

        # Do the backward pass with the gradients
        backward_future = await self.holder.run_coroutine_threadsafe(
            self.forward_backward_async(
                linear_loss_data,
                surrogate_loss_fn,
                None,
            )
        ).result_async()

        # We need to slightly modify the future to add the custom metrics, so we use _CombinedAPIFuture
        # to transform the future.
        def add_custom_metrics(
            results: List[types.ForwardBackwardOutput],
        ) -> types.ForwardBackwardOutput:
            result = results[0]  # Single result
            result.metrics.update(metrics)
            return result

        return _CombinedAPIFuture([backward_future], add_custom_metrics, self.holder)

    @sync_only
    @capture_exceptions(fatal=True)
    def forward_backward_custom_v2(
        self,
        data: List[types.Datum],
        loss_fn: CustomLossFnV2,
        *,
        requested_inputs: Sequence[types.LossInputName] = ("target_logprobs",),
        grouping: types.GroupingSpec | None = None,
        layout: types.Layout = "padded",
    ) -> APIFuture[types.ForwardBackwardOutput]:
        self._assert_custom_loss_v2_supported()
        return self.holder.run_coroutine_threadsafe(
            self.forward_backward_custom_v2_async(
                data,
                loss_fn,
                requested_inputs=requested_inputs,
                grouping=grouping,
                layout=layout,
            )
        ).result()

    @capture_exceptions(fatal=True)
    async def forward_backward_custom_v2_async(
        self,
        data: List[types.Datum],
        loss_fn: CustomLossFnV2,
        *,
        requested_inputs: Sequence[types.LossInputName] = ("target_logprobs",),
        grouping: types.GroupingSpec | None = None,
        layout: types.Layout = "padded",
    ) -> APIFuture[types.ForwardBackwardOutput]:
        await self._assert_custom_loss_v2_supported_async()
        if torch is None:
            raise ImportError("PyTorch is not installed. Cannot run custom forward_backward.")

        requested_inputs_tuple = self._validate_custom_loss_v2_options(requested_inputs, layout)
        if requested_inputs_tuple != ("target_logprobs",):
            raise ValueError(
                "forward_backward_custom_v2 currently supports only "
                "requested_inputs=('target_logprobs',)"
            )
        if not data:
            return _CombinedAPIFuture([], combine_fwd_bwd_output_results, self.holder)

        forward_data = self._get_custom_loss_forward_data(data, "cross_entropy")
        forward_future = await self.holder.run_coroutine_threadsafe(
            self.forward_async(
                forward_data,
                "cross_entropy",
                None,
            )
        ).result_async()
        forward_result = await forward_future.result_async()

        target_logprob_leaves: list[torch.Tensor] = []
        original_shapes: list[tuple[int, ...]] = []
        for out in forward_result.loss_fn_outputs:
            logprob = torch.tensor(out["logprobs"].data)
            if out["logprobs"].shape is not None:
                logprob = logprob.reshape(out["logprobs"].shape)
            logprob = logprob.clone().detach().requires_grad_(True)
            target_logprob_leaves.append(logprob)
            original_shapes.append(tuple(logprob.shape))

        padded_target_logprobs, token_mask, seq_lens = self._build_padded_target_logprob_context(
            target_logprob_leaves
        )

        loss_output = loss_fn(
            types.CustomLossContextV2(
                data=data,
                target_logprobs=padded_target_logprobs,
                token_mask=token_mask,
                seq_lens=seq_lens,
                grouping=grouping,
                metadata={"requested_inputs": list(requested_inputs_tuple)},
                layout=layout,
            )
        )
        if not isinstance(loss_output, types.CustomLossOutputV2):
            raise TypeError(
                "forward_backward_custom_v2 loss_fn must return CustomLossOutputV2; "
                f"got {type(loss_output).__name__}"
            )

        grads = self._extract_target_logprob_grads_for_custom_loss_v2(
            loss_output,
            target_logprob_leaves=target_logprob_leaves,
            padded_target_logprobs=padded_target_logprobs,
            seq_lens=seq_lens,
            original_shapes=original_shapes,
        )

        linear_loss_data = []
        for datum, grad in zip(data, grads, strict=True):
            linear_loss_data.append(
                types.Datum(
                    model_input=datum.model_input,
                    loss_fn_inputs={
                        "target_tokens": datum.loss_fn_inputs["target_tokens"],
                        "weights": -grad,
                    },
                )
            )

        backward_future = await self.holder.run_coroutine_threadsafe(
            self.forward_backward_async(
                linear_loss_data,
                "cross_entropy",
                None,
            )
        ).result_async()

        def add_custom_metrics(
            results: List[types.ForwardBackwardOutput],
        ) -> types.ForwardBackwardOutput:
            result = results[0]
            result.metrics.update(loss_output.metrics)
            return result

        return _CombinedAPIFuture([backward_future], add_custom_metrics, self.holder)

    def _get_custom_loss_forward_data(
        self,
        data: List[types.Datum],
        surrogate_loss_fn: types.LossFnType,
    ) -> List[types.Datum]:
        assert surrogate_loss_fn in _SUPPORTED_CUSTOM_BACKEND_LOSS_FNS, (
            "forward_backward_custom_async should validate surrogate_loss_fn before "
            "_get_custom_loss_forward_data is called"
        )

        forward_data = []
        for datum in data:
            target_tokens = datum.loss_fn_inputs.get("target_tokens")
            if target_tokens is None:
                raise ValueError("target_tokens must be provided when using cross_entropy")

            unexpected_keys = sorted(set(datum.loss_fn_inputs) - {"target_tokens", "weights"})
            if unexpected_keys:
                raise ValueError(
                    "forward_backward_custom only supports loss_fn_inputs keys "
                    "{'target_tokens', 'weights'}; "
                    f"found unexpected keys: {unexpected_keys}"
                )

            if "weights" in datum.loss_fn_inputs:
                forward_data.append(datum)
                continue

            forward_loss_fn_inputs = dict(datum.loss_fn_inputs)
            forward_loss_fn_inputs["weights"] = types.TensorData(
                data=[0.0] * len(target_tokens.data),
                dtype="float32",
                shape=target_tokens.shape,
            )
            forward_data.append(
                types.Datum(
                    model_input=datum.model_input,
                    loss_fn_inputs=forward_loss_fn_inputs,
                )
            )

        return forward_data

    def _assert_custom_loss_v2_supported(self) -> None:
        supports_custom_loss_v2 = None
        if hasattr(self.holder, "server_supports_custom_loss_v2"):
            supports_custom_loss_v2 = self.holder.server_supports_custom_loss_v2()
        if supports_custom_loss_v2 is not True:
            raise ValueError(CUSTOM_LOSS_V2_UNSUPPORTED_ERROR)

    async def _assert_custom_loss_v2_supported_async(self) -> None:
        supports_custom_loss_v2 = None
        if hasattr(self.holder, "server_supports_custom_loss_v2_async"):
            supports_custom_loss_v2 = await self.holder.server_supports_custom_loss_v2_async()
        elif hasattr(self.holder, "server_supports_custom_loss_v2"):
            supports_custom_loss_v2 = self.holder.server_supports_custom_loss_v2()
        if supports_custom_loss_v2 is not True:
            raise ValueError(CUSTOM_LOSS_V2_UNSUPPORTED_ERROR)

    def _validate_custom_loss_v2_options(
        self,
        requested_inputs: Sequence[types.LossInputName],
        layout: types.Layout,
    ) -> tuple[types.LossInputName, ...]:
        requested_inputs_tuple = tuple(requested_inputs)
        if not requested_inputs_tuple:
            raise ValueError("requested_inputs must contain at least one v2 loss input")

        invalid_requested_inputs = sorted(
            {name for name in requested_inputs_tuple if name not in _ALL_CUSTOM_LOSS_V2_INPUTS}
        )
        if invalid_requested_inputs:
            raise ValueError(
                "requested_inputs contains invalid custom loss v2 inputs: "
                f"{invalid_requested_inputs}"
            )

        unsupported_requested_inputs = sorted(
            {
                name
                for name in requested_inputs_tuple
                if name not in _SUPPORTED_CUSTOM_LOSS_V2_INPUTS
            }
        )
        if unsupported_requested_inputs:
            raise ValueError(
                "forward_backward_custom_v2 currently supports only "
                "requested_inputs=('target_logprobs',); "
                f"unsupported inputs: {unsupported_requested_inputs}"
            )

        if layout not in _SUPPORTED_CUSTOM_LOSS_V2_LAYOUTS:
            raise ValueError(
                f"Unsupported custom loss v2 layout={layout!r}. "
                f"Supported layouts are: {sorted(_SUPPORTED_CUSTOM_LOSS_V2_LAYOUTS)}"
            )

        return requested_inputs_tuple

    def _build_padded_target_logprob_context(
        self,
        target_logprob_leaves: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flattened = [logprob.reshape(-1) for logprob in target_logprob_leaves]
        padded = torch.nn.utils.rnn.pad_sequence(
            flattened,
            batch_first=True,
            padding_value=0.0,
        )
        seq_lens = torch.tensor([tensor.numel() for tensor in flattened], dtype=torch.int64)
        token_mask = (
            torch.arange(padded.shape[1], dtype=torch.int64).unsqueeze(0) < seq_lens.unsqueeze(1)
        ).to(dtype=padded.dtype)
        return padded, token_mask, seq_lens

    def _extract_target_logprob_grads_for_custom_loss_v2(
        self,
        loss_output: types.CustomLossOutputV2,
        *,
        target_logprob_leaves: Sequence[torch.Tensor],
        padded_target_logprobs: torch.Tensor,
        seq_lens: torch.Tensor,
        original_shapes: Sequence[tuple[int, ...]],
    ) -> list[torch.Tensor]:
        if loss_output.grad_wrt_inputs is not None:
            unexpected_grad_inputs = sorted(set(loss_output.grad_wrt_inputs) - {"target_logprobs"})
            if unexpected_grad_inputs:
                raise ValueError(
                    "forward_backward_custom_v2 currently supports grad_wrt_inputs "
                    f"only for 'target_logprobs'; found {unexpected_grad_inputs}"
                )
            target_logprob_grads = loss_output.grad_wrt_inputs.get("target_logprobs")
            if target_logprob_grads is None:
                raise ValueError(
                    "forward_backward_custom_v2 grad_wrt_inputs must include "
                    "'target_logprobs' when provided"
                )
            if not isinstance(target_logprob_grads, torch.Tensor):
                target_logprob_grads = torch.tensor(target_logprob_grads)
            target_logprob_grads = (
                target_logprob_grads.detach().clone().to(dtype=padded_target_logprobs.dtype)
            )
            if tuple(target_logprob_grads.shape) != tuple(padded_target_logprobs.shape):
                raise ValueError(
                    "forward_backward_custom_v2 grad_wrt_inputs['target_logprobs'] "
                    f"must match padded target_logprobs shape {tuple(padded_target_logprobs.shape)}; "
                    f"got {tuple(target_logprob_grads.shape)}"
                )
            grads: list[torch.Tensor] = []
            for row_index, (seq_len, original_shape) in enumerate(
                zip(seq_lens.tolist(), original_shapes, strict=True)
            ):
                row = target_logprob_grads[row_index, :seq_len]
                grads.append(row.reshape(original_shape))
            return grads

        loss = loss_output.loss
        if not hasattr(loss, "backward"):
            raise ValueError(
                "forward_backward_custom_v2 loss_fn must return a differentiable loss "
                "or explicit grad_wrt_inputs['target_logprobs']"
            )
        loss.backward()
        grads = []
        for logprob in target_logprob_leaves:
            if logprob.grad is None:
                raise ValueError("No gradient computed for target_logprobs tensor")
            grads.append(logprob.grad)
        return grads

    @capture_exceptions(fatal=True)
    def optim_step(self, adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]:
        """Update model parameters using Adam optimizer.

        The Adam optimizer used by worldlines is identical
        to [torch.optim.AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
        Note that unlike PyTorch, Cybernetics's default weight decay value is 0.0 (no weight decay).


        Args:
        - `adam_params`: Adam optimizer parameters (learning_rate, betas, eps, weight_decay)

        Returns:
        - `APIFuture` containing optimizer step response

        Example:
        ```python
        # First compute gradients
        fwdbwd_future = training_client.forward_backward(data, "cross_entropy")

        # Then update parameters
        optim_future = training_client.optim_step(
            types.AdamParams(
                learning_rate=1e-4,
                weight_decay=0.01
            )
        )

        # Wait for both to complete
        fwdbwd_result = await fwdbwd_future
        optim_result = await optim_future
        ```
        """
        self._assert_training_supported()
        request_id = self._get_request_id()

        @capture_exceptions(fatal=True)
        async def _optim_step_async():
            start_time = time.time()

            async def _send_request():
                request = types.OptimStepRequest(
                    adam_params=adam_params,
                    model_id=self._guaranteed_model_id(),
                    seq_id=request_id + 1,
                )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.training.optim_step(
                        request=request,
                    )

            async with self._take_turn(request_id):
                untyped_future = await self.holder.execute_with_retries(_send_request)
            return await _APIFuture(
                types.OptimStepResponse,
                self.holder,
                untyped_future,
                request_start_time=start_time,
                request_type="OptimStep",
                queue_state_observer=self._queue_state_logger,
            )

        return self.holder.run_coroutine_threadsafe(_optim_step_async())

    async def optim_step_async(
        self, adam_params: types.AdamParams
    ) -> APIFuture[types.OptimStepResponse]:
        """Async version of optim_step."""
        await self._assert_training_supported_async()

        async def _optim_step_on_holder() -> APIFuture[types.OptimStepResponse]:
            request_id = self._get_request_id()
            start_time = time.time()

            async def _send_request():
                request = types.OptimStepRequest(
                    adam_params=adam_params,
                    model_id=self._guaranteed_model_id(),
                    seq_id=request_id + 1,
                )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.training.optim_step(
                        request=request,
                    )

            async with self._take_turn(request_id):
                untyped_future = await self.holder.execute_with_retries(_send_request)
            return _APIFuture(
                types.OptimStepResponse,
                self.holder,
                untyped_future,
                request_start_time=start_time,
                request_type="OptimStep",
                queue_state_observer=self._queue_state_logger,
            )

        return await self.holder.run_coroutine_threadsafe(_optim_step_on_holder()).result_async()

    @capture_exceptions(fatal=True)
    def save_state(
        self, name: str, ttl_seconds: int | None = None
    ) -> APIFuture[types.SaveWeightsResponse]:
        """Save model weights to persistent storage.

        Args:
        - `name`: Name for the saved checkpoint
        - `ttl_seconds`: Optional TTL in seconds for the checkpoint (None = never expires)

        Returns:
        - `APIFuture` containing the save response with checkpoint path

        Example:
        ```python
        # Save after training
        save_future = training_client.save_state("checkpoint-001")
        result = await save_future
        print(f"Saved to: {result.path}")
        ```
        """
        request_id = self._get_request_id()

        @capture_exceptions(fatal=True)
        async def _save_state_async():
            start_time = time.time()

            async def _send_request():
                request = types.SaveWeightsRequest(
                    model_id=self._guaranteed_model_id(),
                    path=name,
                    seq_id=request_id + 1,
                    ttl_seconds=ttl_seconds,
                )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.save(
                        request=request,
                    )

            async with self._take_turn(request_id):
                future = await self.holder.execute_with_retries(_send_request)
            return await _APIFuture(
                types.SaveWeightsResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="SaveWeights",
                queue_state_observer=self._queue_state_logger,
            )

        return self.holder.run_coroutine_threadsafe(_save_state_async())

    async def save_state_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> APIFuture[types.SaveWeightsResponse]:
        """Async version of save_state."""
        return self.save_state(name, ttl_seconds=ttl_seconds)

    @capture_exceptions(fatal=True)
    async def _load_state_impl(
        self, request_id: int, path: str, optimizer: bool
    ) -> types.LoadWeightsResponse:
        start_time = time.time()

        async def _send_request():
            request = types.LoadWeightsRequest(
                model_id=self._guaranteed_model_id(),
                path=path,
                seq_id=request_id + 1,
                optimizer=optimizer,
            )
            with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                return await client.weights.load(
                    request=request,
                )

        async with self._take_turn(request_id):
            future = await self.holder.execute_with_retries(_send_request)
        return await _APIFuture(
            types.LoadWeightsResponse,
            self.holder,
            future,
            request_start_time=start_time,
            request_type="LoadWeights",
            queue_state_observer=self._queue_state_logger,
        )

    @capture_exceptions(fatal=True)
    def load_state(self, path: str) -> APIFuture[types.LoadWeightsResponse]:
        """Load model weights from a saved checkpoint.

        This loads only the model weights, not optimizer state (e.g., Adam momentum).
        To also restore optimizer state, use load_state_with_optimizer.

        Args:
        - `path`: Cybernetics path to saved weights (e.g., "worldlines://run-id/weights/checkpoint-001")

        Returns:
        - `APIFuture` containing the load response

        Example:
        ```python
        # Load checkpoint to continue training (weights only, optimizer resets)
        load_future = training_client.load_state("worldlines://run-id/weights/checkpoint-001")
        await load_future
        # Continue training from loaded state
        ```
        """
        request_id = self._get_request_id()
        return self.holder.run_coroutine_threadsafe(self._load_state_impl(request_id, path, False))

    async def load_state_async(self, path: str) -> APIFuture[types.LoadWeightsResponse]:
        """Async version of load_state."""
        return self.load_state(path)

    @capture_exceptions(fatal=True)
    def load_state_with_optimizer(self, path: str) -> APIFuture[types.LoadWeightsResponse]:
        """Load model weights and optimizer state from a checkpoint.

        Args:
        - `path`: Cybernetics path to saved weights (e.g., "worldlines://run-id/weights/checkpoint-001")

        Returns:
        - `APIFuture` containing the load response

        Example:
        ```python
        # Resume training with optimizer state
        load_future = training_client.load_state_with_optimizer(
            "worldlines://run-id/weights/checkpoint-001"
        )
        await load_future
        # Continue training with restored optimizer momentum
        ```
        """
        self._assert_optimizer_restore_supported()
        request_id = self._get_request_id()
        return self.holder.run_coroutine_threadsafe(self._load_state_impl(request_id, path, True))

    async def load_state_with_optimizer_async(
        self, path: str
    ) -> APIFuture[types.LoadWeightsResponse]:
        """Async version of load_state_with_optimizer."""
        await self._assert_optimizer_restore_supported_async()

        async def _load_state_with_optimizer_on_holder() -> APIFuture[types.LoadWeightsResponse]:
            request_id = self._get_request_id()
            start_time = time.time()

            async def _send_request():
                request = types.LoadWeightsRequest(
                    model_id=self._guaranteed_model_id(),
                    path=path,
                    seq_id=request_id + 1,
                    optimizer=True,
                )
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    return await client.weights.load(
                        request=request,
                    )

            async with self._take_turn(request_id):
                future = await self.holder.execute_with_retries(_send_request)
            return _APIFuture(
                types.LoadWeightsResponse,
                self.holder,
                future,
                request_start_time=start_time,
                request_type="LoadWeights",
                queue_state_observer=self._queue_state_logger,
            )

        return await self.holder.run_coroutine_threadsafe(
            _load_state_with_optimizer_on_holder()
        ).result_async()

    @capture_exceptions(fatal=True)
    async def _save_weights_for_sampler_impl(
        self, request_id: int, name: str | None, ttl_seconds: int | None = None
    ) -> types.SaveWeightsForSamplerResponseInternal:
        assert asyncio.get_event_loop() == self.holder.get_loop()
        start_time = time.time()

        async def _send_request():
            if name is not None:
                request = types.SaveWeightsForSamplerRequest(
                    model_id=self._guaranteed_model_id(),
                    path=name,
                    seq_id=request_id + 1,
                    ttl_seconds=ttl_seconds,
                )
            else:
                # Training client can never be created from a shadow holder, so we can safely assert
                assert self.holder._sampling_client_counter is not None
                sampling_session_seq_id = self.holder._sampling_client_counter
                self.holder._sampling_client_counter += 1
                request = types.SaveWeightsForSamplerRequest(
                    model_id=self._guaranteed_model_id(),
                    seq_id=request_id + 1,
                    sampling_session_seq_id=sampling_session_seq_id,
                    ttl_seconds=ttl_seconds,
                )
            with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                return await client.weights.save_for_sampler(
                    request=request,
                )

        async with self._take_turn(request_id):
            future = await self.holder.execute_with_retries(_send_request)
        return await _APIFuture(
            types.SaveWeightsForSamplerResponseInternal,
            self.holder,
            future,
            request_start_time=start_time,
            request_type="SaveWeightsForSampler",
            queue_state_observer=self._queue_state_logger,
        )

    @capture_exceptions(fatal=True)
    def save_weights_for_sampler(
        self, name: str, ttl_seconds: int | None = None
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        """Save model weights for use with a SamplingClient.

        Args:
        - `name`: Name for the saved sampler weights
        - `ttl_seconds`: Optional TTL in seconds for the checkpoint (None = never expires)

        Returns:
        - `APIFuture` containing the save response with sampler path

        Example:
        ```python
        # Save weights for inference
        save_future = training_client.save_weights_for_sampler("sampler-001")
        result = await save_future
        print(f"Sampler weights saved to: {result.path}")

        # Use the path to create a sampling client
        sampling_client = service_client.create_sampling_client(
            model_path=result.path
        )
        ```
        """
        request_id = self._get_request_id()

        async def _save_weights_for_sampler_async():
            result = await self._save_weights_for_sampler_impl(request_id, name, ttl_seconds)
            assert result.path is not None
            return types.SaveWeightsForSamplerResponse(
                path=result.path,
                has_optimizer_state=result.has_optimizer_state,
            )

        return self.holder.run_coroutine_threadsafe(_save_weights_for_sampler_async())

    async def save_weights_for_sampler_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        """Async version of save_weights_for_sampler."""
        return self.save_weights_for_sampler(name, ttl_seconds=ttl_seconds)

    def _get_info_submit(self) -> AwaitableConcurrentFuture[types.GetInfoResponse]:
        async def _get_info_async():
            async def _send_request():
                with self.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
                    request = types.GetInfoRequest(model_id=self._guaranteed_model_id())
                    return await client.models.get_info(
                        request=request,
                    )

            return await self.holder.execute_with_retries(_send_request)

        return self.holder.run_coroutine_threadsafe(_get_info_async())

    @sync_only
    @capture_exceptions(fatal=True)
    def get_info(self) -> types.GetInfoResponse:
        """Get information about the current model.

        Returns:
        - `GetInfoResponse` with model configuration and metadata

        Example:
        ```python
        info = training_client.get_info()
        print(f"Model ID: {info.model_data.model_id}")
        print(f"Base model: {info.model_data.model_name}")
        print(f"LoRA rank: {info.model_data.lora_rank}")
        ```
        """
        return self._get_info_submit().result()

    @capture_exceptions(fatal=True)
    async def get_info_async(self) -> types.GetInfoResponse:
        """Async version of get_info."""
        return await self._get_info_submit()

    @capture_exceptions(fatal=True)
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer for the current model.

        Returns:
        - `PreTrainedTokenizer` compatible with the model

        Example:
        ```python
        tokenizer = training_client.get_tokenizer()
        tokens = tokenizer.encode("Hello world")
        text = tokenizer.decode(tokens)
        ```
        """
        return _get_tokenizer(self._guaranteed_model_id(), self.holder)

    @capture_exceptions(fatal=True)
    def create_sampling_client(
        self, model_path: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        """Create a SamplingClient from saved weights.

        Args:
        - `model_path`: Cybernetics path to saved weights
        - `retry_config`: Optional configuration for retrying failed requests

        Returns:
        - `SamplingClient` configured with the specified weights

        Example:
        ```python
        sampling_client = training_client.create_sampling_client(
            "worldlines://run-id/weights/checkpoint-001"
        )
        # Use sampling_client for inference
        ```
        """
        return SamplingClient.create(
            self.holder, model_path=model_path, retry_config=retry_config
        ).result()

    @capture_exceptions(fatal=True)
    async def create_sampling_client_async(
        self, model_path: str, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        """Async version of create_sampling_client."""
        return await SamplingClient.create(
            self.holder, model_path=model_path, retry_config=retry_config
        )

    def save_weights_and_get_sampling_client_submit(
        self,
        retry_config: RetryConfig | None = None,
        name: str | None = None,
    ) -> APIFuture[SamplingClient]:
        request_id = self._get_request_id()

        async def _save_weights_and_get_sampling_client_async():
            result = await self._save_weights_for_sampler_impl(request_id, name)
            if result.path is not None:
                return await SamplingClient.create(
                    self.holder,
                    model_path=result.path,
                    retry_config=retry_config,
                )
            assert result.sampling_session_id is not None
            return await SamplingClient.create(
                self.holder,
                sampling_session_id=result.sampling_session_id,
                retry_config=retry_config,
            )

        return self.holder.run_coroutine_threadsafe(_save_weights_and_get_sampling_client_async())

    @capture_exceptions(fatal=True)
    def save_weights_and_get_sampling_client(
        self, name: str | None = None, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        """Save current weights and create a SamplingClient for inference.

        Args:
        - `name`: Optional name for the saved weights. When provided, the helper
          creates a persistent named sampler checkpoint and binds the returned sampler
          to that `worldlines://...` path. When omitted, the helper uses the existing
          ephemeral hidden-checkpoint flow.
        - `retry_config`: Optional configuration for retrying failed requests

        Returns:
        - `SamplingClient` configured with the current model weights

        Example:
        ```python
        # After training, create a sampling client directly
        sampling_client = training_client.save_weights_and_get_sampling_client()

        # Now use it for inference
        prompt = types.ModelInput.from_ints(tokenizer.encode("Hello"))
        params = types.SamplingParams(max_tokens=20)
        result = sampling_client.sample(prompt, 1, params).result()
        ```
        """
        return self.save_weights_and_get_sampling_client_submit(
            retry_config,
            name=name,
        ).result()

    @capture_exceptions(fatal=True)
    async def save_weights_and_get_sampling_client_async(
        self, name: str | None = None, retry_config: RetryConfig | None = None
    ) -> SamplingClient:
        """Async version of save_weights_and_get_sampling_client."""
        return await self.save_weights_and_get_sampling_client_submit(
            retry_config,
            name=name,
        )

    def get_telemetry(self) -> Telemetry | None:
        return self.holder.get_telemetry()


def _get_tokenizer(model_id: types.ModelID, holder: InternalClientHolder) -> PreTrainedTokenizer:
    """Get tokenizer for a training model by fetching model info first."""

    async def _get_info_async():
        with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            request = types.GetInfoRequest(model_id=model_id)
            return await client.models.get_info(request=request)

    info = holder.run_coroutine_threadsafe(_get_info_async()).result()
    model_name = info.model_data.model_name
    assert model_name is not None, "This shouldn't happen: model_name is None"

    return _load_tokenizer_from_model_info(model_name, info.model_data.tokenizer_id)
