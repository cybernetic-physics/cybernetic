# Client API

Exhaustive reference for the five hand-written, supported client classes in the
Cybernetics SDK: `ServiceClient`, `TrainingClient`, `SamplingClient`,
`RestClient`, and `APIFuture`. Every signature, parameter default, and return
type below is copied from the source under
`src/cybernetics/lib/public_interfaces/`.

For the conceptual picture of how these clients fit together, see the
[mental model](../concepts/mental-model.md). For task recipes, see
[Training](../guides/training.md), [Sampling & Inference](../guides/sampling.md),
and [Checkpoints](../guides/checkpoints.md). Response field definitions live in
[Data Contracts](../reference/data-contracts.md).

## What is and isn't public

Install the package as `cybernetic-physics`; import it as `cybernetics`:

```bash
pip install cybernetic-physics
```

```python
import cybernetics
from cybernetics import ServiceClient, TrainingClient, SamplingClient, APIFuture
from cybernetics import types
```

Four classes are re-exported at the package root: `ServiceClient`,
`TrainingClient`, `SamplingClient`, and `APIFuture`. `RestClient` is **not**
re-exported — you obtain it from `service_client.create_rest_client()`.

These five classes are the entire supported surface. The Stainless-generated
`AsyncCybernetics` client and the `cybernetics.resources.*` modules are the
**internal transport** that these clients sit on top of. Do not import or call
them directly: they are unstable, undocumented here, and not covered by any
compatibility guarantee. If you find yourself reaching into
`cybernetics.resources`, you are off the blessed path.

This page documents only public methods. Underscore-prefixed methods
(`_create_impl`, `_get_info_submit`, etc.) are internal and omitted.

## Brand vocabulary you will see in signatures

The product is **Cybernetics**. **Worldlines** is a legacy name that survives in
two places you will hit in this API:

- **`worldlines://` paths.** Saved-weight and checkpoint paths look like
  `worldlines://<run-id>/weights/<checkpoint-id>`. This string format is current
  and correct — it is what `save_state`, `load_state`, and the
  `*_from_worldlines_path` REST methods consume and produce.
- **`session_id`.** `ServiceClient.session_id` is described in source as the
  "Worldlines session ID backing this client."

**CP** stands for **Cybernetic Physics**, the billing/auth brand. It appears in
the `CP_API_KEY` environment variable and in `cp_live_` API key prefixes. See
[Authentication](../guides/authentication.md) for the full key-resolution order
(explicit → `CYBERNETICS_API_KEY` → `CP_API_KEY` → deprecated
`WORLDLINES_API_KEY` → stored login).

## Return types: three shapes

Methods return one of three things. Knowing which determines how you wait for a
result.

| Return shape | How to get the value | Awaitable? |
|---|---|---|
| `APIFuture[T]` | `f.result()` or `await f` | Yes — directly |
| `concurrent.futures.Future[T]` | `f.result()` | Not directly; wrap with `AwaitableConcurrentFuture(f)` to await |
| direct value `T` | already resolved | n/a |

`APIFuture[T]` is the richer object (see [APIFuture](#apifuture)). A plain
`concurrent.futures.Future` is the standard-library future; you can only block
on it with `.result()` unless you wrap it.

Each method's exact shape is marked in the tables below.

## The `_async` twin convention (global rule)

Almost every public method has an `_async` twin: the same name with `_async`
appended (`forward` → `forward_async`, `get_training_run` →
`get_training_run_async`). The twin is a coroutine — `await` it instead of
calling the sync method when you are already inside an event loop. The tables
below list each method once; assume the twin exists unless noted otherwise.

What awaiting the twin gives you mirrors the sync return shape, with one wrinkle
worth internalizing:

- **Data-plane training methods** (`forward`, `forward_backward`,
  `forward_backward_custom`, `optim_step`, `save_state`, `load_state`,
  `load_state_with_optimizer`, `save_weights_for_sampler`) return an
  `APIFuture[T]` from *both* forms. The sync method hands you an `APIFuture` you
  call `.result()` on; the `_async` twin is a coroutine whose awaited value is
  *also* an `APIFuture` you must await a second time. Example:

  ```python
  # sync
  fut = training_client.forward_backward(data, "cross_entropy")
  out = fut.result()

  # async — note the double await
  fut = await training_client.forward_backward_async(data, "cross_entropy")
  out = await fut
  ```

- **Read / control methods** (`get_info`, `get_server_capabilities`,
  `get_training_run`, `sample`, `get_base_model`, ...) resolve to the value
  directly when you await the twin — no second await.

A few methods have **no** `_async` twin: the `session_id` property,
`create_rest_client`, `get_telemetry`, and the already-async-shaped
`save_weights_and_get_sampling_client_submit`.

### `@sync_only` is a warning, not a guard

Several sync methods are decorated `@sync_only`. If called from inside a running
asyncio event loop (outside Jupyter), they **log a warning** pointing you at the
`_async` twin and then **run anyway** — they do not raise. Calling sync methods
from async code risks deadlocks and stalls, so heed the warning and switch to the
twin. (Source: `cybernetics/lib/sync_only.py`.)

---

## ServiceClient

`cybernetics.ServiceClient` — the main entry point. Construct one, then use it to
mint `TrainingClient`, `SamplingClient`, and `RestClient` instances.

```python
class ServiceClient(
    user_metadata: dict[str, str] | None = None,
    project_id: str | None = None,
    sample_dispatch_profile: str = "auto",
    **kwargs: Any,
)
```

Constructing a `ServiceClient` is near-instant. `**kwargs` are forwarded to the
underlying HTTP client (API keys, headers, connection settings); auth headers are
resolved automatically from the environment if not supplied (see
[Authentication](../guides/authentication.md)).

### Properties

| Member | Type | Notes |
|---|---|---|
| `session_id` | `str` | Worldlines session ID backing this client. Property, no `_async` twin. |

### Methods

| Method | Returns | Shape |
|---|---|---|
| `get_server_capabilities()` | `types.GetServerCapabilitiesResponse` | direct value |
| `get_scheduler_state()` | `types.GetSchedulerStateResponse` | direct value |
| `create_lora_training_client(...)` | `TrainingClient` | direct value |
| `create_training_client_from_state(path, user_metadata=None)` | `TrainingClient` | direct value |
| `create_training_client_from_state_with_optimizer(path, user_metadata=None)` | `TrainingClient` | direct value |
| `create_sampling_client(...)` | `SamplingClient` | direct value |
| `create_rest_client()` | `RestClient` | direct value, **no `_async` twin** |
| `get_telemetry()` | `Telemetry | None` | direct value, no `_async` twin |

Full signatures for the methods that take more than a path:

```python
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
) -> TrainingClient
```

At least one of `train_mlp`, `train_attn`, `train_unembed` must be `True` (an
`assert` enforces this). `timeout` is the seconds to wait for model
creation/cold-start. This call takes a moment — it allocates resources and warms
the model.

```python
def create_sampling_client(
    self,
    model_path: str | None = None,
    base_model: str | None = None,
    retry_config: RetryConfig | None = None,
    timeout: float | None = None,
) -> SamplingClient
```

Raises `ValueError` if **neither** `model_path` nor `base_model` is provided. If
**both** are provided, `model_path` (the checkpoint) wins and the backend infers
the base model from the checkpoint.

`create_training_client_from_state` restores **weights only** (optimizer state
resets). `create_training_client_from_state_with_optimizer` restores weights
**and** optimizer state (e.g. Adam momentum), and raises `ValueError` if the
backend does not support optimizer restore or if the checkpoint at `path` has no
optimizer state. Both take a `worldlines://...` path.

```python
service_client = cybernetics.ServiceClient()

training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B", rank=16,
)
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
rest_client = service_client.create_rest_client()
```

---

## TrainingClient

`cybernetics.TrainingClient` — one fine-tuned model you can train and sample
from. You get one from `service_client.create_lora_training_client()` or one of
the `create_training_client_from_state*` helpers. **Do not** construct it
directly; the `__init__` signature
(`TrainingClient(holder, model_seq_id, model_id)`) takes an internal holder.

### Methods

| Method | Returns | Shape |
|---|---|---|
| `forward(data, loss_fn, loss_fn_config=None)` | `APIFuture[types.ForwardBackwardOutput]` | APIFuture |
| `forward_backward(data, loss_fn, loss_fn_config=None)` | `APIFuture[types.ForwardBackwardOutput]` | APIFuture |
| `forward_backward_custom(data, loss_fn, *, loss_type_input="logprobs")` | `APIFuture[types.ForwardBackwardOutput]` | APIFuture |
| `forward_backward_custom_v2(data, loss_fn, *, requested_inputs=("target_logprobs",), grouping=None, layout="padded")` | `APIFuture[types.ForwardBackwardOutput]` | APIFuture |
| `optim_step(adam_params)` | `APIFuture[types.OptimStepResponse]` | APIFuture |
| `save_state(name, ttl_seconds=None)` | `APIFuture[types.SaveWeightsResponse]` | APIFuture |
| `load_state(path)` | `APIFuture[types.LoadWeightsResponse]` | APIFuture |
| `load_state_with_optimizer(path)` | `APIFuture[types.LoadWeightsResponse]` | APIFuture |
| `save_weights_for_sampler(name, ttl_seconds=None)` | `APIFuture[types.SaveWeightsForSamplerResponse]` | APIFuture |
| `save_weights_and_get_sampling_client_submit(retry_config=None, name=None)` | `APIFuture[SamplingClient]` | APIFuture (this *is* the async-shaped form; no `_async` twin) |
| `save_weights_and_get_sampling_client(name=None, retry_config=None)` | `SamplingClient` | direct value |
| `create_sampling_client(model_path, retry_config=None)` | `SamplingClient` | direct value |
| `get_info()` | `types.GetInfoResponse` | direct value |
| `get_tokenizer()` | `PreTrainedTokenizer` | direct value, no `_async` twin |
| `get_telemetry()` | `Telemetry | None` | direct value, no `_async` twin |

Type signatures for the data-plane methods:

```python
def forward(
    self,
    data: List[types.Datum],
    loss_fn: types.LossFnType,
    loss_fn_config: Dict[str, float] | None = None,
) -> APIFuture[types.ForwardBackwardOutput]

def forward_backward(
    self,
    data: List[types.Datum],
    loss_fn: types.LossFnType,
    loss_fn_config: Dict[str, float] | None = None,
) -> APIFuture[types.ForwardBackwardOutput]

def optim_step(
    self, adam_params: types.AdamParams
) -> APIFuture[types.OptimStepResponse]
```

`forward`, `forward_backward`, and `optim_step` raise `ValueError` if the backend
is inference-only (does not support training operations). The Adam optimizer
matches `torch.optim.AdamW`, **except** the default `weight_decay` is `0.0` (no
weight decay) rather than PyTorch's default.

`save_state` and `save_weights_for_sampler` accept `ttl_seconds: int | None`
(None = never expires). `save_state` persists a training checkpoint; reload it
with `load_state` / `load_state_with_optimizer`. `save_weights_for_sampler`
persists weights for inference and returns a path you feed to
`create_sampling_client`.

`load_state_with_optimizer` raises `ValueError` if the backend does not support
optimizer-state restore.

`save_weights_and_get_sampling_client` is the convenience path after training:
when `name` is `None` it uses an ephemeral hidden checkpoint; when `name` is set
it creates a persistent named sampler checkpoint bound to a `worldlines://...`
path.

```python
fwdbwd_future = training_client.forward_backward(data, "cross_entropy")
optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
fwdbwd_result = fwdbwd_future.result()   # ForwardBackwardOutput
optim_result = optim_future.result()     # OptimStepResponse

sampling_client = training_client.save_weights_and_get_sampling_client("my-model")
```

### Custom loss functions

```python
CustomLossFnV1 = Callable[[List[types.Datum], List[Any]], Tuple[Any, Dict[str, float]]]
CustomLossFnV2 = Callable[[types.CustomLossContextV2], types.CustomLossOutputV2]

def forward_backward_custom(
    self,
    data: List[types.Datum],
    loss_fn: CustomLossFnV1,
    *,
    loss_type_input: Literal["logprobs"] = "logprobs",
) -> APIFuture[types.ForwardBackwardOutput]

def forward_backward_custom_v2(
    self,
    data: List[types.Datum],
    loss_fn: CustomLossFnV2,
    *,
    requested_inputs: Sequence[types.LossInputName] = ("target_logprobs",),
    grouping: types.GroupingSpec | None = None,
    layout: types.Layout = "padded",
) -> APIFuture[types.ForwardBackwardOutput]
```

Honest limits, straight from source:

- Both require PyTorch installed (`ImportError` otherwise). `forward_backward_custom`
  only supports `loss_fn_inputs` keys `{"target_tokens", "weights"}` and
  `loss_type_input="logprobs"` (the only supported value).
- `forward_backward_custom_v2` is **gated**: it raises if the backend does not
  report `supports_custom_loss_v2`. Check
  `get_server_capabilities().supports_custom_loss_v2` first.
- Despite the broad `requested_inputs` type, v2 today supports **only**
  `requested_inputs=("target_logprobs",)` and **only** `layout="padded"`.
  Anything else raises `ValueError`. The remaining input names
  (`advantages`, `returns`, `values`, etc.) and layouts are defined in the type
  but not yet implemented.
- `forward_backward_custom_v2` has no docstring in source; it is the newer,
  narrower API.

---

## SamplingClient

`cybernetics.SamplingClient` — text generation / inference from a base model or
saved weights. Obtain one from `service_client.create_sampling_client()` or
`training_client.save_weights_and_get_sampling_client()`. Do not construct
directly; `__init__` takes an internal holder and a keyword-only
`sampling_session_id`.

**Sampling is training-first today.** The clean, supported way to get a
`SamplingClient` is from a `ServiceClient`/`TrainingClient`; the examples and
flows in this SDK are built around the train-then-sample loop, not a standalone
inference product.

### Note on return shape

The two hot-path methods, `sample` and `compute_logprobs`, return a plain
**`concurrent.futures.Future`**, *not* an `APIFuture`. Call `.result()` to block.
To await one, wrap it: `await AwaitableConcurrentFuture(future)` — which is
exactly what the `_async` twins do, and the twins resolve to the value directly.

### Methods

| Method | Returns | Shape |
|---|---|---|
| `sample(prompt, num_samples, sampling_params, include_prompt_logprobs=False, topk_prompt_logprobs=0)` | `concurrent.futures.Future[types.SampleResponse]` | concurrent Future |
| `compute_logprobs(prompt)` | `concurrent.futures.Future[list[float | None]]` | concurrent Future |
| `get_base_model()` | `str` | direct value |
| `get_tokenizer()` | `PreTrainedTokenizer` | direct value, no `_async` twin |
| `get_telemetry()` | `Telemetry | None` | direct value, no `_async` twin |
| `create(holder, *, model_path=None, base_model=None, sampling_session_id=None, retry_config=None)` | `APIFuture[SamplingClient]` | static factory, called internally by the `ServiceClient`/`TrainingClient` helpers |

Signatures:

```python
def sample(
    self,
    prompt: types.ModelInput,
    num_samples: int,
    sampling_params: types.SamplingParams,
    include_prompt_logprobs: bool = False,
    topk_prompt_logprobs: int = 0,
) -> concurrent.futures.Future[types.SampleResponse]

def compute_logprobs(
    self, prompt: types.ModelInput
) -> concurrent.futures.Future[list[float | None]]
```

`compute_logprobs` returns one log-prob per prompt token; `None` entries mark
tokens where a log-prob could not be computed.

```python
prompt = types.ModelInput.from_ints(tokenizer.encode("The weather today is"))
params = types.SamplingParams(max_tokens=20, temperature=0.7)
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result = future.result()
for sample in result.samples:
    print(tokenizer.decode(sample.tokens))
```

### Multiprocessing and subprocess isolation

`SamplingClient` is picklable and safe to pass to multiple worker
processes — but always create it in the **main** process and pass it down;
`ServiceClient` and `TrainingClient` must stay in the main process.

Set `CYBERNETICS_SUBPROCESS_SAMPLING=1` to run `sample()` and
`compute_logprobs()` in a dedicated subprocess, so CPU-heavy user code (grading,
environment stepping) does not stall networking and heartbeats via GIL
contention. It is transparent — the same API works with or without it.

`get_tokenizer()` requires the optional `transformers` dependency; if missing it
raises `ImportError` telling you to
`pip install 'cybernetic-physics[tokenizers]'`.

---

## RestClient

`cybernetics.lib...RestClient` — metadata and checkpoint management over REST.
Obtain it from `service_client.create_rest_client()`. Not re-exported at the
package root. `__init__` takes the internal holder only.

### Return-shape caveat

Most `RestClient` methods return a plain **`concurrent.futures.Future`** (block
with `.result()`). **Two exceptions** return an `APIFuture` (which you can
`await` directly): `get_weights_info_by_worldlines_path` and `get_sampler`. The
table marks each.

`access_scope: Literal["owned", "accessible"] = "owned"` recurs across the read
methods: `"owned"` restricts to runs/sessions you own; `"accessible"` widens to
everything you can reach.

### Methods

| Method | Returns | Shape |
|---|---|---|
| `get_training_run(training_run_id, access_scope="owned")` | `Future[types.TrainingRun]` | concurrent Future |
| `get_training_run_by_worldlines_path(worldlines_path, access_scope="owned")` | `Future[types.TrainingRun]` | concurrent Future |
| `get_weights_info_by_worldlines_path(worldlines_path)` | `APIFuture[types.WeightsInfoResponse]` | **APIFuture** |
| `list_training_runs(limit=20, offset=0, access_scope="owned")` | `Future[types.TrainingRunsResponse]` | concurrent Future |
| `list_checkpoints(training_run_id)` | `Future[types.CheckpointsListResponse]` | concurrent Future |
| `list_user_checkpoints(limit=100, offset=0)` | `Future[types.CheckpointsListResponse]` | concurrent Future |
| `get_checkpoint_archive_url(training_run_id, checkpoint_id)` | `Future[types.CheckpointArchiveUrlResponse]` | concurrent Future |
| `get_checkpoint_archive_url_from_worldlines_path(worldlines_path)` | `Future[types.CheckpointArchiveUrlResponse]` | concurrent Future |
| `delete_checkpoint(training_run_id, checkpoint_id)` | `Future[None]` | concurrent Future |
| `delete_checkpoint_from_worldlines_path(worldlines_path)` | `Future[None]` | concurrent Future |
| `publish_checkpoint_from_worldlines_path(worldlines_path)` | `Future[None]` | concurrent Future |
| `unpublish_checkpoint_from_worldlines_path(worldlines_path)` | `Future[None]` | concurrent Future |
| `set_checkpoint_ttl_from_worldlines_path(worldlines_path, ttl_seconds)` | `Future[None]` | concurrent Future |
| `get_session(session_id, access_scope="owned")` | `Future[types.GetSessionResponse]` | concurrent Future |
| `list_sessions(limit=20, offset=0, access_scope="owned")` | `Future[types.ListSessionsResponse]` | concurrent Future |
| `cancel_session(session_id)` | `Future[types.WorldlinesSessionCancelResponse]` | concurrent Future |
| `get_sampler(sampler_id)` | `APIFuture[types.GetSamplerResponse]` | **APIFuture** |
| `get_telemetry()` | `Telemetry | None` | direct value, no `_async` twin |

Notes from source:

- `set_checkpoint_ttl_from_worldlines_path(worldlines_path, ttl_seconds)` —
  `ttl_seconds: int | None`. A positive int expires the checkpoint that many
  seconds from now; `None` removes any existing TTL.
- `publish` / `unpublish` only work for the **exact owner** of the training run.
  Documented HTTP failure modes: `400` invalid checkpoint identifier; `404` not
  found or not owner; `409` already public (publish) / already private
  (unpublish); `500` server error. TTL errors add `400` when `ttl_seconds <= 0`.
- `get_checkpoint_archive_url*` may take a while; the SDK logs a warning while it
  builds the archive. The response carries a signed `url` and `expires_at`; you
  download the archive yourself with any HTTP client.
- `cancel_session` requests safe cleanup of the session's compute lease; the
  control plane only stops session-owned leases when nothing else references them.

```python
rest_client = service_client.create_rest_client()

run = rest_client.get_training_run("run-id").result()
print(run.training_run_id, run.base_model, run.is_lora)

ckpts = rest_client.list_checkpoints("run-id").result()
for c in ckpts.checkpoints:
    print(c.checkpoint_type, c.checkpoint_id)

# APIFuture — awaitable directly
info = await rest_client.get_weights_info_by_worldlines_path(
    "worldlines://run-id/weights/checkpoint-001"
)
print(info.base_model, info.lora_rank)
```

Field-by-field definitions of `TrainingRun`, `CheckpointsListResponse`,
`WeightsInfoResponse`, `GetSessionResponse`, etc. are in
[Data Contracts](../reference/data-contracts.md). For failure-code
troubleshooting see [Errors & Troubleshooting](../reference/errors.md).

---

## APIFuture

`cybernetics.APIFuture` — abstract base class (`Generic[T]`) for results that can
be used both synchronously and asynchronously. It is what the data-plane
`TrainingClient` methods and a couple of `RestClient` methods return.

```python
class APIFuture(ABC, Generic[T]):
    async def result_async(self, timeout: float | None = None) -> T: ...
    def result(self, timeout: float | None = None) -> T: ...
    def __await__(self): ...   # awaiting the future == await self.result_async()
```

| Method | Returns | Notes |
|---|---|---|
| `result(timeout=None)` | `T` | Blocks until complete. `timeout` in seconds; `None` waits indefinitely. Raises `TimeoutError` on timeout. |
| `result_async(timeout=None)` | `T` (coroutine) | Async form; same timeout semantics. |
| `__await__()` | — | Lets you write `await future` directly; equivalent to `await future.result_async()`. |

Because `__await__` delegates to `result_async()`, an `APIFuture` is awaitable as
a value:

```python
future = training_client.forward_backward(data, "cross_entropy")
result = future.result()       # sync, blocks
# result = await future        # async, same result
```

### AwaitableConcurrentFuture

`AwaitableConcurrentFuture(future)` is the concrete `APIFuture` implementation
that wraps a `concurrent.futures.Future`. Use it to `await` a plain future
returned by `SamplingClient.sample` / `compute_logprobs` or the
concurrent-future `RestClient` methods.

```python
from cybernetics.lib.public_interfaces.api_future import AwaitableConcurrentFuture

class AwaitableConcurrentFuture(APIFuture[T]):
    def __init__(self, future: concurrent.futures.Future[T]): ...
    def result(self, timeout: float | None = None) -> T: ...
    async def result_async(self, timeout: float | None = None) -> T: ...
    def future(self) -> concurrent.futures.Future[T]: ...
```

| Method | Returns | Notes |
|---|---|---|
| `result(timeout=None)` | `T` | Delegates to the wrapped future's `result`. |
| `result_async(timeout=None)` | `T` (coroutine) | Wraps the future via `asyncio.wrap_future` under an `asyncio.timeout`. |
| `future()` | `concurrent.futures.Future[T]` | Returns the underlying std-lib future (e.g. to poll `.done()`). |

```python
# Await a SamplingClient result without using sample_async:
result = await AwaitableConcurrentFuture(
    sampling_client.sample(prompt, 1, types.SamplingParams(max_tokens=20))
)
```

---

## See also

- [Quickstart](../quickstart.md) — first end-to-end run.
- [Training](../guides/training.md) and [SFT vs RL](../concepts/sft-vs-rl.md).
- [Sampling & Inference](../guides/sampling.md).
- [Checkpoints](../guides/checkpoints.md) — `save_state`, `worldlines://` paths,
  publish/TTL.
- [Data Contracts](../reference/data-contracts.md) — request/response field
  definitions.
- [Errors & Troubleshooting](../reference/errors.md) — HTTP codes and gated
  features.
- [Glossary](../concepts/glossary.md) — Cybernetics / Worldlines / CP and other
  terms.
