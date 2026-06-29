# Data Contracts

This page is the exhaustive reference for the [pydantic](https://docs.pydantic.dev/) models the SDK serializes onto (and off of) the wire. When you build a training batch, request a sample, or configure an optimizer through the [blessed clients](./client-api.md), you are constructing instances of these models. Reach for this page when you need the exact field name, type, default, or coercion rule.

Every model documented here is importable from the package root:

```python
from cybernetics import (
    Datum,
    ModelInput,
    EncodedTextChunk,
    TensorData,
    SamplingParams,
    AdamParams,
    OptimStepRequest,
    Checkpoint,
    StopReason,
    TensorDtype,
)
```

They also live under `cybernetics.types` (e.g. `from cybernetics.types import LossFnType`). Both import paths resolve to the same classes.

> The install package is `cybernetic-physics`; the import package is `cybernetics`. See the [quickstart](../quickstart.md) for `pip install cybernetic-physics`. The `worldlines://` strings you will see in [Checkpoint](#checkpoint) are the legacy "Worldlines" name surviving in storage paths — see the [glossary](../concepts/glossary.md) and [authentication guide](../guides/authentication.md) for the Cybernetics / Worldlines / CP naming history.

---

## The two base classes: StrictBase vs BaseModel

Every contract on this page inherits from one of two bases defined in `cybernetics/_models.py`. The choice encodes whether the model is something **you send** or something **the server sends back**.

| Base class | `model_config` | Used for | Behavior on unknown fields |
|---|---|---|---|
| `StrictBase` | `ConfigDict(frozen=True, extra="forbid")` | **Request** types | Raises `ValidationError` |
| `BaseModel` | `ConfigDict(frozen=True, extra="ignore")` | **Response** types | Silently dropped |

Two consequences worth internalizing:

- **`extra="forbid"` on requests means typos fail loudly.** If you pass `temperatur=0.7` to a request model you will get a validation error at construction time, not a silently-ignored field. This is deliberate: it catches client bugs before they reach the backend.
- **`extra="ignore"` on responses means forward compatibility.** When the server starts returning a new field, your installed (older) client ignores it instead of crashing. So a [Checkpoint](#checkpoint) you receive may carry fields not listed here; do not rely on their absence.

**All models are frozen.** `frozen=True` makes every instance immutable — assigning to a field after construction raises a `ValidationError`. The helper methods that look mutating (e.g. `ModelInput.append`) return a **new** instance rather than mutating in place. Treat these objects as values, not records.

```python
from cybernetics import SamplingParams

sp = SamplingParams(temperature=0.7)
sp.temperature = 0.9
# pydantic_core._pydantic_core.ValidationError: 1 validation error for SamplingParams
# temperature
#   Instance is frozen [type=frozen_instance, ...]
```

```python
from cybernetics import SamplingParams

SamplingParams(temperatur=0.7)   # note the typo — but SamplingParams is a *response-shaped*
                                 # BaseModel (extra="ignore"), so this is silently dropped.
                                 # Request models like Datum would instead raise.
```

> Note: `SamplingParams` inherits from the lenient `BaseModel` (`extra="ignore"`), not `StrictBase`. This is a deliberate exception — see [its section](#samplingparams). Request-side models like `Datum`, `ModelInput`, `TensorData`, `AdamParams`, and `OptimStepRequest` all inherit from `StrictBase` and reject unknown fields.

---

## Datum

`cybernetics.Datum` — `StrictBase`. The unit of a training batch: one model input plus the per-token tensors a loss function consumes. Used by the [TrainingClient](./client-api.md) forward-backward path. See the [training guide](../guides/training.md) for how a list of `Datum` becomes a step.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `model_input` | [`ModelInput`](#modelinput) | required | The tokens (and/or image chunks) fed to the model for this example. |
| `loss_fn_inputs` | `Dict[str, TensorData]` | required | "Dictionary mapping field names to tensor data" — the per-token targets/weights the loss reads (e.g. `target_tokens`, `weights`, `advantages`, `logprobs`). |

### Automatic tensor conversion

`Datum` runs a `model_validator(mode="before")` that converts the **values** of `loss_fn_inputs` into [`TensorData`](#tensordata) for you. You may pass any of:

- a `torch.Tensor` → converted via `TensorData.from_torch` (requires torch installed),
- a `numpy.ndarray` → converted via `TensorData.from_numpy`,
- a plain Python `list` of numbers → converted to numpy, then to `TensorData`,
- an already-built `TensorData` → passed through unchanged.

```python
import numpy as np
from cybernetics import Datum, ModelInput

datum = Datum(
    model_input=ModelInput.from_ints([1, 2, 3, 4]),
    loss_fn_inputs={
        "target_tokens": [2, 3, 4, 5],          # list -> int64 TensorData
        "weights": np.ones(4, dtype=np.float32), # ndarray -> float32 TensorData
    },
)
```

The dtype a bare Python list is coerced to is **keyed by the field name**, via this fixed table in `datum.py`:

| `loss_fn_inputs` key | Coerced dtype |
|---|---|
| `target_tokens` | `int64` |
| `weights` | `float32` |
| `advantages` | `float32` |
| `logprobs` | `float32` |
| `clip_low_threshold` | `float32` |
| `clip_high_threshold` | `float32` |

A key **not** in this table that is passed as a bare list will raise `KeyError` during conversion — pass a `numpy`/`torch` array (whose dtype is read directly) or a pre-built `TensorData` for any other key.

**Ragged lists are rejected.** If you pass a nested list that is not rectangular, conversion raises:

```text
ValueError: target_tokens must be a rectangular numeric array; ragged nested lists are not supported
```

The exact loss functions and which `loss_fn_inputs` keys each one reads are out of scope here; see [SFT vs RL](../concepts/sft-vs-rl.md) and the [LossFnType](#lossfntype) enum below.

---

## ModelInput

`cybernetics.ModelInput` — `StrictBase`. An ordered sequence of input chunks. Formerly called `TokenSequence`.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `chunks` | `List[ModelInputChunk]` | required | "Sequence of input chunks (formerly TokenSequence)" |

`ModelInputChunk` is a discriminated union (on the `type` field) of three chunk models:

| Chunk type | `type` literal | Key fields |
|---|---|---|
| `EncodedTextChunk` | `"encoded_text"` | `tokens: Sequence[int]` — "Array of token IDs" |
| `ImageChunk` | `"image"` | `data: bytes` (base64 on the wire), `format: "png" \| "jpeg"`, `expected_tokens: int \| None` |
| `ImageAssetPointerChunk` | `"image_asset_pointer"` | `location: str` (path/URL), `format: "png" \| "jpeg"`, `expected_tokens: int \| None` |

For image chunks, `expected_tokens` is **advisory only**: the backend computes the real token count from the image and fails fast if it disagrees with `expected_tokens`. Reading `.length` on an image chunk whose `expected_tokens` is `None` raises `ValueError`.

### Helper methods

`ModelInput` is the most ergonomic of the contracts because most callers only ever use text tokens. All "mutating" helpers return a new frozen instance.

| Method | Signature | Behavior |
|---|---|---|
| `ModelInput.from_ints` | `(tokens: List[int]) -> ModelInput` | Wraps a token list in a single `EncodedTextChunk`. |
| `ModelInput.empty` | `() -> ModelInput` | An input with no chunks. |
| `.to_ints` | `() -> List[int]` | Flattens all chunks back to a token list. **Raises `ValueError` if any chunk is not an `EncodedTextChunk`.** |
| `.length` | property `-> int` | Total context length = sum of each chunk's `.length`. |
| `.append` | `(chunk: ModelInputChunk) -> ModelInput` | Returns a new `ModelInput` with the chunk appended. |
| `.append_int` | `(token: int) -> ModelInput` | Appends a one-token `EncodedTextChunk`. |

```python
from cybernetics import ModelInput

mi = ModelInput.from_ints([10, 11, 12])
mi = mi.append_int(13)        # new instance; original unchanged
mi.to_ints()                  # [10, 11, 12, 13]
mi.length                     # 4
```

---

## TensorData

`cybernetics.TensorData` — `StrictBase`. The wire representation of a tensor: a flat list of numbers plus a dtype and an optional shape. This is what `loss_fn_inputs` values become.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `data` | `Union[List[int], List[float]]` | required | "Flattened tensor data as array of numbers." Row-major flattened. |
| `dtype` | [`TensorDtype`](#tensordtype) | required | `"int64"` or `"float32"`. |
| `shape` | `Optional[List[int]]` | `None` | "The shape of the tensor (see PyTorch tensor.shape). The shape of a one-dimensional list of length N is `(N,)`. Can usually be inferred if not provided, and is generally inferred as a 1D tensor." |

### Constructors and converters

| Method | Signature | Notes |
|---|---|---|
| `TensorData.from_numpy` | `(array: np.ndarray) -> TensorData` | Flattens, maps dtype, records `array.shape`. |
| `TensorData.from_torch` | `(tensor: torch.Tensor) -> TensorData` | Same, for torch. Requires torch installed. |
| `.to_numpy` | `() -> np.ndarray` | Rebuilds and reshapes (if `shape` set). |
| `.to_torch` | `() -> torch.Tensor` | Raises `ImportError` if torch is not installed. |
| `.tolist` | `() -> List[Any]` | `self.to_numpy().tolist()` — a nested Python list respecting `shape`. |

### Honest warning: dtype coercion is lossy

`TensorData` only carries two dtypes (`int64`, `float32`). The conversion functions in `tensor_data.py` map **every** floating input to `float32` and **every** integer input to `int64` by inspecting the dtype *kind*, not its width:

- `_convert_numpy_dtype_to_tensor`: `kind == "f"` → `"float32"`, `kind == "i"` → `"int64"`, anything else raises `ValueError`.
- `_convert_torch_dtype_to_tensor`: floating-point dtypes → `"float32"`, everything else → `"int64"`.

Concretely, this means:

- A `float64` numpy array round-tripped through `TensorData.from_numpy(...).to_numpy()` comes back as **`float32`** — you lose precision. There is no way to preserve `float64` in this contract.
- A `torch.bfloat16` or `torch.float16` tensor is widened/narrowed to `float32`.
- Unsigned numpy integer arrays (`kind == "u"`) raise `ValueError` from `from_numpy` (only signed `kind == "i"` is accepted). Note that `Datum`'s list-conversion path accepts `kind in ("f", "i", "u")` and casts to `int64`/`float32`, so unsigned data passed as a *bare list to a `loss_fn_inputs` key* is handled — but passed as a numpy array it is not.

```python
import numpy as np
from cybernetics import TensorData

td = TensorData.from_numpy(np.array([1.0, 2.0], dtype=np.float64))
td.dtype                     # 'float32'  <-- coerced, not float64
td.to_numpy().dtype          # dtype('float32')
```

If you need to preserve precision wider than `float32`, this contract cannot do it today. Quantize or scale your data deliberately before constructing `TensorData`.

---

## SamplingParams

`cybernetics.SamplingParams` — **`BaseModel`** (`extra="ignore"`, not `StrictBase`). Controls generation for the [SamplingClient](./client-api.md). See the [sampling guide](../guides/sampling.md).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `max_tokens` | `Optional[int]` | `None` | "Maximum number of tokens to generate" |
| `seed` | `Optional[int]` | `None` | "Random seed for reproducible generation" |
| `stop` | `Union[str, Sequence[str], Sequence[int], None]` | `None` | "Stop sequences for generation" — a string, a list of strings, or a list of token IDs. |
| `temperature` | `float` | `1` | "Sampling temperature" |
| `top_k` | `int` | `-1` | "Top-k sampling parameter (-1 for no limit)" |
| `top_p` | `float` | `1` | "Nucleus sampling probability" |
| `json_schema` | `Optional[Dict[str, Any]]` | `None` | "Optional JSON schema for grammar-constrained decoding (forwarded to sglang's xgrammar backend as `sampling_params.json_schema`). When set, the model is forced to emit a token sequence that, when decoded, parses as a JSON document conforming to this schema." |
| `response_format` | `Optional[Dict[str, Any]]` | `None` | OpenAI-shaped alias. Accepts `{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {"schema": {...}}}`. If `json_schema` is `None` and a schema is present here, it is auto-extracted. Held separately so clients can round-trip the original shape. |
| `completion_logprobs` | `bool` | `False` | "Return per-token logprobs for the completion (each entry in `SampledSequence.logprobs`). Only the picked-token logprobs are returned, not the full distribution. Off by default to save bandwidth." |

### response_format → json_schema unwrapping

A `model_validator(mode="before")` normalizes `response_format` into `json_schema` when `json_schema` is not already set:

- `{"type": "json_schema", "json_schema": {"schema": {...}}}` → the inner `schema` is copied into `json_schema`.
- `{"type": "json_object"}` → `json_schema` becomes `{"type": "object"}`.

So you can supply either field; the model keeps both, with `json_schema` taking precedence if you set it explicitly.

```python
from cybernetics import SamplingParams

sp = SamplingParams(
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
    response_format={"type": "json_object"},
)
sp.json_schema   # {'type': 'object'}  (auto-derived from response_format)
```

> Honest scope note: sampling is training-first today. The fields above are the full set the SDK serializes; `json_schema`/`response_format` are forwarded to the sglang xgrammar backend and the SDK does not validate your schema for you.

---

## AdamParams

`cybernetics.AdamParams` — `StrictBase`. Optimizer hyperparameters for an optimizer step. See the [training guide](../guides/training.md).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `learning_rate` | `float` | `0.0001` | "Learning rate for the optimizer" |
| `beta1` | `float` | `0.9` | "Coefficient used for computing running averages of gradient" |
| `beta2` | `float` | `0.95` | "Coefficient used for computing running averages of gradient square" |
| `eps` | `float` | `1e-12` | "Term added to the denominator to improve numerical stability" |
| `weight_decay` | `float` | `0.0` | "Weight decay for the optimizer. Uses decoupled weight decay." |
| `grad_clip_norm` | `float` | `0.0` | "Maximum global gradient norm. If the global gradient norm is greater than this value, it will be clipped to this value. 0.0 means no clipping." |

### OptimStepRequest

`AdamParams` is carried inside `cybernetics.OptimStepRequest` — `StrictBase`, the message that asks the backend to apply an optimizer step.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `adam_params` | `AdamParams` | required | The optimizer hyperparameters above. |
| `model_id` | `ModelID` (`str`) | required | Which model/run to step. |
| `seq_id` | `Optional[int]` | `None` | Ordering/sequence id for the step. |
| `type` | `Literal["optim_step"]` | `"optim_step"` | Discriminator; do not override. |

> `OptimStepRequest` sets `protected_namespaces=()` so that a `model_`-prefixed field name (`model_id`) does not collide with pydantic's reserved `model_` namespace.

```python
from cybernetics import AdamParams, OptimStepRequest

req = OptimStepRequest(
    model_id="my-run-id",
    adam_params=AdamParams(learning_rate=3e-4, weight_decay=0.01),
)
```

---

## Checkpoint

`cybernetics.Checkpoint` — **`BaseModel`** (response type, `extra="ignore"`). Returned when you list or create checkpoints. See the [checkpoints guide](../guides/checkpoints.md).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `checkpoint_id` | `str` | required | "The checkpoint ID" |
| `checkpoint_type` | `CheckpointType` (`"training" \| "sampler"`) | required | "The type of checkpoint (training or sampler)" |
| `time` | `datetime` | required | "The time when the checkpoint was created" |
| `worldlines_path` | `str` | required | "The worldlines path to the checkpoint" — a `worldlines://...` string (legacy storage namespace). |
| `size_bytes` | `int \| None` | `None` | "The size of the checkpoint in bytes" |
| `public` | `bool` | `False` | "Whether the checkpoint is publicly accessible" |
| `expires_at` | `datetime \| None` | `None` | "When this checkpoint expires (None = never expires)" |

### ParsedCheckpointCyberneticsPath

A companion `BaseModel` for decomposing a `worldlines://` path. `ParsedCheckpointCyberneticsPath.from_worldlines_path(path)` parses a string of the form `worldlines://<training_run_id>/<weights|sampler_weights>/<...>` into:

| Field | Type | Meaning |
|---|---|---|
| `worldlines_path` | `str` | The original path. |
| `training_run_id` | `str` | First path segment. |
| `checkpoint_type` | `CheckpointType` | `"training"` if the segment is `weights`, `"sampler"` if `sampler_weights`. |
| `checkpoint_id` | `str` | The remaining segments joined by `/`. |

It raises `ValueError` if the string does not start with `worldlines://`, has fewer than 3 segments, or the second segment is neither `weights` nor `sampler_weights`.

```python
from cybernetics import ParsedCheckpointCyberneticsPath

p = ParsedCheckpointCyberneticsPath.from_worldlines_path(
    "worldlines://run-42/weights/step-1000"
)
p.training_run_id   # 'run-42'
p.checkpoint_type   # 'training'
p.checkpoint_id     # 'weights/step-1000'
```

---

## Enumerations

These are `Literal` type aliases, not enum classes — you pass the bare string. Construction validates against the allowed set.

### TensorDtype

`Literal["int64", "float32"]` — the only two tensor element types the wire format supports.

| Value | Meaning |
|---|---|
| `"int64"` | 64-bit signed integers. Maps to `np.int64` / `torch.int64`. The dtype for token-id tensors (e.g. `target_tokens`). |
| `"float32"` | 32-bit floats. Maps to `np.float32` / `torch.float32`. The dtype for weights, advantages, logprobs, thresholds. |

There is no `float64`, `float16`, `bfloat16`, or unsigned variant. See the [TensorData coercion warning](#honest-warning-dtype-coercion-is-lossy).

### StopReason

`Literal["length", "stop"]` — why a sampled sequence ended.

| Value | Meaning |
|---|---|
| `"length"` | Generation hit the token budget (`max_tokens`) without emitting a stop sequence. |
| `"stop"` | Generation halted because it produced a configured stop sequence / end token. |

### LossFnType

`Literal["cross_entropy", "importance_sampling", "ppo", "cispo", "dro", "flow_rwr"]` — the loss functions the training backend supports.

| Value | Meaning |
|---|---|
| `"cross_entropy"` | Standard supervised cross-entropy. The SFT workhorse — see [SFT vs RL](../concepts/sft-vs-rl.md). |
| `"importance_sampling"` | Importance-sampling-weighted policy-gradient loss. |
| `"ppo"` | Proximal Policy Optimization clipped objective. |
| `"cispo"` | Clipped importance-sampling policy optimization variant. |
| `"dro"` | Distributionally-robust optimization objective. |
| `"flow_rwr"` | Flow reward-weighted regression objective. |

> Honest scope note: the source defines this enum as the set of accepted values; it does not document each loss's exact math or its required `loss_fn_inputs` keys. The names above are described from the contract, not from a per-loss spec. Confirm the inputs a given loss needs in the [training guide](../guides/training.md) before relying on it.

---

## Don't construct the internal transport directly

You build the models on this page and hand them to the [blessed clients](./client-api.md) — `ServiceClient`, `TrainingClient`, `SamplingClient`, and `APIFuture` (re-exported at the package root from `cybernetics/lib/public_interfaces/`). Those clients own serialization.

The Stainless-generated `AsyncCybernetics` / `cybernetics.resources.*` layer is the **internal HTTP transport**. It is not the supported surface — do not call it directly. If a contract here serializes incorrectly through a high-level client, that is a bug to report, not a reason to drop down to the generated resources.

## See also

- [Client API](./client-api.md) — the methods that consume these contracts.
- [Mental model](../concepts/mental-model.md) — how datums, steps, and checkpoints fit together.
- [Training guide](../guides/training.md) and [Sampling guide](../guides/sampling.md) — task recipes.
- [Errors & Troubleshooting](./errors.md) — what validation failures look like in practice.
