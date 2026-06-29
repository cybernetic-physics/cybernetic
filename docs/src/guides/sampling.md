# Sampling & Inference

This is the how-to for **using** a model: turning a base model or a set of
trained weights into generated tokens. If you have just finished a training
loop and want to see what your model actually does, you are in the right place.

One honest caveat up front: the SDK ships almost entirely *training-first*. The
only sampling examples that exist today live in docstrings, and one of them is
wrong (it iterates `result.samples`, a field that does not exist — the real
field is `result.sequences`; see [Reading the response](#reading-the-response)).
Everything below is written against the actual signatures and field names in
the source, so treat this page as the first correct end-to-end reference.

For the conceptual picture of where sampling sits relative to training, see the
[mental model](../concepts/mental-model.md). For the exhaustive type listing,
see [data contracts](../reference/data-contracts.md) and the
[client API reference](../reference/client-api.md).

## Prerequisites

```bash
pip install cybernetic-physics
```

The install name is `cybernetic-physics`; the import package is `cybernetics`.
You also need credentials resolvable by the SDK — set one of
`CYBERNETICS_API_KEY`, `CP_API_KEY`, or the deprecated `WORLDLINES_API_KEY`
(the three brand names — Cybernetics, CP for *Cybernetic Physics*, and the
legacy *Worldlines* — all coexist; see [authentication](./authentication.md)
for the full resolution order).

`get_tokenizer()` and any encode/decode step need the optional tokenizer
dependency:

```bash
pip install 'cybernetic-physics[tokenizers]'
```

Everything here uses the blessed high-level clients — `ServiceClient`,
`TrainingClient`, `SamplingClient` — re-exported from the package root. Do **not**
reach for `AsyncCybernetics` or `cybernetics.resources.*`; that is the internal
transport layer, not a supported surface.

## Step 1 — Get a SamplingClient

There are three supported ways to obtain a `SamplingClient`. Pick the one that
matches what you already have.

### A. From a base model (no training)

The fastest path. Sample directly from a stock base model:

```python
from cybernetics import ServiceClient, types

service_client = ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
```

`create_sampling_client` is synchronous and blocks until the sampler is ready
(cold start can take a moment; pass `timeout=<seconds>` to cap the wait). Its
full signature:

```python
def create_sampling_client(
    self,
    model_path: str | None = None,
    base_model: str | None = None,
    retry_config: RetryConfig | None = None,
    timeout: float | None = None,
) -> SamplingClient: ...
```

You must pass `model_path` **or** `base_model`. Passing neither raises
`ValueError("Either model_path or base_model must be provided")`. If you pass
**both**, the checkpoint path wins and the `base_model` hint is dropped — the
backend infers the base model from the checkpoint.

### B. From saved weights (a checkpoint path)

If you already saved weights and have a `worldlines://...` path (these paths
keep the legacy *Worldlines* scheme — see [checkpoints](./checkpoints.md)),
point `model_path` at it:

```python
sampling_client = service_client.create_sampling_client(
    model_path="worldlines://run-id/weights/checkpoint-001"
)
```

### C. Straight from a TrainingClient

This is the common end-of-training move. `TrainingClient` exposes a one-call
helper that saves the current weights and hands you back a wired-up sampler:

```python
def save_weights_and_get_sampling_client(
    self, name: str | None = None, retry_config: RetryConfig | None = None
) -> SamplingClient: ...
```

```python
# ... after forward_backward / optim_step (see the training guide) ...
sampling_client = training_client.save_weights_and_get_sampling_client()
```

- Omit `name` to use an **ephemeral hidden checkpoint** — convenient for a quick
  eval right after training, not durable.
- Pass `name="my-eval-snapshot"` to create a **persistent named checkpoint** and
  bind the returned sampler to its `worldlines://...` path.

Two lower-level variants exist if you want to manage the checkpoint yourself:

```python
# Save, then wire up separately. save_weights_for_sampler returns an APIFuture.
save_result = training_client.save_weights_for_sampler("sampler-001").result()
sampling_client = training_client.create_sampling_client(save_result.path)
```

`RetryConfig` (optional everywhere above) lives at
`cybernetics.lib.retry_handler.RetryConfig`; its defaults are sensible, so you
rarely need it. See [client API](../reference/client-api.md) for its fields.

> Multi-process note: `SamplingClient` is picklable and safe to share across
> worker processes. Always create it (and the `ServiceClient`/`TrainingClient`)
> in the main process, then pass the sampler to workers — never construct the
> service/training clients in a child process.

## Step 2 — Build the prompt as a ModelInput

`sample()` does not take a string. It takes a `ModelInput`, which is a sequence
of token chunks. The simplest constructor wraps a flat list of token IDs:

```python
tokenizer = sampling_client.get_tokenizer()  # requires the [tokenizers] extra
prompt = types.ModelInput.from_ints(tokenizer.encode("The weather today is"))
```

`ModelInput` (re-exported as `cybernetics.ModelInput` and `types.ModelInput`)
gives you:

| Method / property | Meaning |
|---|---|
| `ModelInput.from_ints(tokens: list[int])` | Build from a flat token-ID list (wraps them in one `EncodedTextChunk`). |
| `ModelInput.empty()` | An empty input (`chunks == []`). |
| `.append(chunk)` | Return a new `ModelInput` with one more chunk. |
| `.append_int(token)` | Return a new `ModelInput` with one more token appended. |
| `.to_ints()` | Flatten back to `list[int]`. Raises `ValueError` if any chunk is not an `EncodedTextChunk` (e.g. an image chunk). |
| `.length` | Total context length (token count) across all chunks. |

Under the hood a `ModelInput.chunks` list holds `EncodedTextChunk`
(`type="encoded_text"`, `tokens=[...]`) and — for multimodal models — image
chunks. For text generation, `from_ints` is all you need.

## Step 3 — Configure sampling with SamplingParams

`SamplingParams` (`cybernetics.SamplingParams` / `types.SamplingParams`) is a
plain pydantic model. Every field and its real default, copied from source:

| Field | Type | Default | Notes |
|---|---|---|---|
| `max_tokens` | `int \| None` | `None` | Max tokens to generate. **Set this** — leaving it `None` means no explicit cap (the code carries a `TODO` to make it required). |
| `temperature` | `float` | `1` | Sampling temperature. `0` ≈ greedy. |
| `top_k` | `int` | `-1` | Top-k cutoff; `-1` means no limit. |
| `top_p` | `float` | `1` | Nucleus sampling probability; `1` means no limit. |
| `seed` | `int \| None` | `None` | Seed for reproducible generation. |
| `stop` | `str \| Sequence[str] \| Sequence[int] \| None` | `None` | Stop sequence(s), as text or token IDs. |
| `json_schema` | `dict \| None` | `None` | Grammar-constrained decoding: forces output that decodes to JSON matching this schema (forwarded to sglang's xgrammar backend). |
| `response_format` | `dict \| None` | `None` | OpenAI-shaped alias. Accepts `{"type": "json_object"}` or `{"type": "json_schema", "json_schema": {"schema": {...}}}`; when `json_schema` is unset, the schema is auto-extracted from here. |
| `completion_logprobs` | `bool` | `False` | When `True`, populate per-token `logprobs` on each returned `SampledSequence` (only the picked-token logprob, not the full distribution). Off by default to save bandwidth. |

```python
params = types.SamplingParams(max_tokens=64, temperature=0.7, top_p=0.95)
```

## Step 4 — Call sample() and read the result

```python
def sample(
    self,
    prompt: types.ModelInput,
    num_samples: int,
    sampling_params: types.SamplingParams,
    include_prompt_logprobs: bool = False,
    topk_prompt_logprobs: int = 0,
) -> concurrent.futures.Future[types.SampleResponse]: ...
```

`sample()` returns immediately with a `concurrent.futures.Future`. Call
`.result()` to block for the response. (This is a plain `Future`, not the
`APIFuture` returned by training calls — there is no `.result_async()` on it;
for async code use `sample_async(...)` instead, covered
[below](#async-usage).)

```python
future = sampling_client.sample(prompt, num_samples=2, sampling_params=params)
response = future.result()
```

### Reading the response

`SampleResponse` has these fields:

| Field | Type | Notes |
|---|---|---|
| `sequences` | `Sequence[SampledSequence]` | One entry per requested sample. **This is the field to iterate** — not `samples`. |
| `type` | `Literal["sample"]` | Always `"sample"`. |
| `prompt_logprobs` | `list[float \| None] \| None` | Populated only when `include_prompt_logprobs=True`. |
| `topk_prompt_logprobs` | `list[list[tuple[int, float]] \| None] \| None` | Populated only when `topk_prompt_logprobs=k > 0`; per prompt token, up to k `(token_id, logprob)` tuples. |

Each `SampledSequence`:

| Field | Type | Notes |
|---|---|---|
| `tokens` | `list[int]` | Generated token IDs. Decode them with the tokenizer. |
| `stop_reason` | `Literal["length", "stop"]` | `"length"` = hit `max_tokens`; `"stop"` = stopped on an end/stop condition. |
| `logprobs` | `list[float] \| None` | Per-token completion logprobs; non-`None` only when `completion_logprobs=True` was set in `SamplingParams`. |

```python
for seq in response.sequences:
    text = tokenizer.decode(seq.tokens)
    print(f"[{seq.stop_reason}] {text}")
```

> Source accuracy note: the `SamplingClient.sample` docstring shows
> `for sample in result.samples`. There is **no** `samples` attribute on
> `SampleResponse` — that example will raise `AttributeError`. Use
> `response.sequences`, as shown above.

## Full end-to-end example

Putting Steps 1–4 together — the first complete, correct sampling script:

```python
from cybernetics import ServiceClient, types

# 1. Get a sampler (base model here; swap in model_path= for trained weights)
service_client = ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")

# 2. Tokenize and build the prompt
tokenizer = sampling_client.get_tokenizer()
prompt = types.ModelInput.from_ints(tokenizer.encode("The weather today is"))

# 3. Configure generation
params = types.SamplingParams(max_tokens=64, temperature=0.7, top_p=0.95)

# 4. Sample (returns a concurrent.futures.Future) and read sequences
response = sampling_client.sample(prompt, num_samples=2, sampling_params=params).result()

for i, seq in enumerate(response.sequences):
    print(f"--- sample {i} (stop_reason={seq.stop_reason}) ---")
    print(tokenizer.decode(seq.tokens))
```

Expected shape of the result (token IDs and text will vary by model):

```text
--- sample 0 (stop_reason=length) ---
 sunny with a light breeze, and the forecast suggests ...
--- sample 1 (stop_reason=stop) ---
 expected to be mostly clear.
```

## Scoring text: compute_logprobs

To score an existing sequence rather than generate, use `compute_logprobs`. It
returns one log-probability per prompt token (with `None` where a value could
not be computed — typically the first position, which has no preceding context):

```python
def compute_logprobs(
    self, prompt: types.ModelInput
) -> concurrent.futures.Future[list[float | None]]: ...
```

```python
prompt = types.ModelInput.from_ints(tokenizer.encode("Hello world"))
logprobs = sampling_client.compute_logprobs(prompt).result()

for i, lp in enumerate(logprobs):
    if lp is not None:
        print(f"token {i}: logprob = {lp:.4f}")
```

Mechanically this is a `sample()` with `num_samples=1`, `max_tokens=1`, and
`include_prompt_logprobs=True`, returning the response's `prompt_logprobs`. If
you want both generation *and* prompt logprobs in a single round trip, call
`sample(..., include_prompt_logprobs=True)` directly and read both
`response.sequences` and `response.prompt_logprobs`.

## Inspecting the model

```python
sampling_client.get_base_model()   # -> str, e.g. "Qwen/Qwen3-8B"
sampling_client.get_tokenizer()    # -> transformers.PreTrainedTokenizer
```

`get_tokenizer()` needs the `[tokenizers]` extra. Without it you get:

```text
ImportError: get_tokenizer() requires the optional 'transformers' dependency.
Install it with: pip install 'cybernetic-physics[tokenizers]'
```

## Async usage

Every blocking call has an `async` sibling that awaits the response directly
(no `.result()`):

```python
response = await sampling_client.sample_async(prompt, 1, params)
logprobs = await sampling_client.compute_logprobs_async(prompt)
base_model = await sampling_client.get_base_model_async()
```

`sample_async` / `compute_logprobs_async` take the same arguments as their sync
counterparts and return the value directly.

## Failure modes and tuning

- **Sampling paused (queue backpressure).** When the backend throttles you
  (HTTP 429), the SDK does not raise — it backs off and retries internally, and
  logs a warning roughly once a minute:

  ```text
  WARNING Sampling is paused for sampler <id>. Reason: concurrent sampler weights limit hit
  ```

  Other reasons you may see: `"Cybernetics backend is running short on capacity,
  please wait"`. Your `.result()` call simply takes longer; no action needed
  beyond patience or reducing concurrency. See
  [errors & troubleshooting](../reference/errors.md).

- **`max_tokens` unset.** Leaving `max_tokens=None` does not error today, but
  the field is on track to become required. Always set it explicitly.

- **`to_ints()` on a non-text input.** Calling `ModelInput.to_ints()` when the
  input contains image chunks raises `ValueError`. `to_ints` is text-only.

- **CPU-heavy callers stalling IO.** If your process does heavy CPU work
  between samples (grading, environment stepping) and you see heartbeats or
  networking stall, set `CYBERNETICS_SUBPROCESS_SAMPLING=1` to run `sample()`
  and `compute_logprobs()` in a dedicated subprocess. The API is unchanged with
  or without it.

## Related pages

- [Quickstart](../quickstart.md) — end-to-end first run.
- [Training](./training.md) — produce the weights you sample from.
- [Checkpoints](./checkpoints.md) — `worldlines://` paths, saving, TTLs.
- [Behavior CI](./behavior-ci.md) — wire sampling into automated checks.
- [Client API reference](../reference/client-api.md) — full method/field tables.
- [Data contracts](../reference/data-contracts.md) — `SamplingParams`,
  `ModelInput`, `SampleResponse`, `SampledSequence` in full.
