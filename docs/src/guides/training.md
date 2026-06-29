# Training

This is a how-to for running a **real training loop** against the Cybernetics
SDK: building batches of `Datum`, calling `forward_backward(...)` to accumulate
gradients, applying `optim_step(...)` to update weights, and checkpointing with
`save_state` / `load_state`. It assumes you already have a `ServiceClient` and a
`TrainingClient` — if not, start with the [quickstart](../quickstart.md) and
[authentication](./authentication.md).

> Install is `pip install cybernetic-physics`; the import package is
> `cybernetics`. Names you will see: **Cybernetics** is the product/SDK,
> **Worldlines** is the legacy name that survives in `worldlines://` checkpoint
> paths and the deprecated `WORLDLINES_API_KEY`, and **CP** ("Cybernetic
> Physics") shows up in `CP_API_KEY` / `cp_live_` keys. They all refer to the
> same platform.

Everything here uses the hand-written high-level clients — `ServiceClient`,
`TrainingClient`, `SamplingClient`, `APIFuture` — re-exported at the package
root (`cybernetics.ServiceClient`, `cybernetics.types`). The
Stainless-generated `AsyncCybernetics` / `cybernetics.resources.*` layer is the
internal transport. **Do not call it directly**; it is not the supported
surface and its shapes change.

---

## What "training" means here

Cybernetics training is an **imperative loop you drive from your process**. There
is no `fit()` and no managed trainer. You repeatedly:

1. Build a list of `types.Datum` (one batch).
2. Call `training.forward_backward(data, loss_fn)` — returns an `APIFuture`. The
   backend runs the forward and backward passes and accumulates gradients.
3. Call `training.optim_step(types.AdamParams(...))` — returns an `APIFuture`.
   The backend applies one Adam/AdamW update and **resets the accumulated
   gradients**.
4. Wait on the futures (`.result()`), read `loss` / `metrics`, repeat.

`forward_backward` followed by `optim_step` is one optimizer step. Loop over your
data to run many.

For **which loss function** to pass (SFT cross-entropy vs the RL losses), see
[SFT vs RL](../concepts/sft-vs-rl.md). The `loss_fn` argument is a
`types.LossFnType`, a `Literal` of exactly these values (from
`src/cybernetics/types/loss_fn_type.py`):

```text
cross_entropy   importance_sampling   ppo   cispo   dro   flow_rwr
```

This page uses `"cross_entropy"` (supervised fine-tuning). Always confirm the
backend advertises the family you want before you spend GPU time:

```python
caps = client.get_server_capabilities()
if "cross_entropy" not in set(caps.loss_families or []):
    raise SystemExit(f"backend loss_families={caps.loss_families}")
```

---

## Honesty about the shipped examples

The two examples in `examples/` are **single-step smokes**, not full training
loops. Read them as the canonical shape, not as a finished training script:

- `examples/dreamzero_sft_smoke.py` — builds **one synthetic** DROID batch,
  encodes it to a `Datum`, and (only with `--remote-run`) does exactly one
  `forward_backward` → `optim_step` → `save_state`. Default mode is local-only
  and just prints the wire keys; it never touches GPU.
- `examples/finetune_on_real_data.py` — a **template**, not a runnable script.
  It requires `lerobot` and a real `LeRobotDataset`, and the entire
  state/action ↔ collate mapping is marked `# TODO: confirm ...` against your
  dataset's feature schema (joint count, image keys, fps). The `--dataset`
  default is a **placeholder**. It also does a single step.

So: the SDK ships the *single step* verified end to end. The *loop* and the
*real dataset wiring* are yours to write. Below is the realistic shape of both,
grounded in those examples — without inventing dataset specifics we cannot see.

---

## Building a `Datum`

A batch is a `List[types.Datum]`. Each `Datum` (from
`src/cybernetics/types/datum.py`) has two fields:

- `model_input: types.ModelInput` — the input sequence.
- `loss_fn_inputs: Dict[str, TensorData]` — the per-loss targets/weights.

For **text** inputs, build `model_input` from token ids with
`ModelInput.from_ints(...)`. `Datum` auto-converts numpy arrays, torch tensors,
and numeric lists in `loss_fn_inputs` into `TensorData`, so you can pass raw
arrays. For `cross_entropy`, the expected key is `target_tokens` (and optionally
`weights`):

```python
from cybernetics import types

tokenizer = training.get_tokenizer()

prompt_ids = tokenizer.encode("Translate to French: hello")
target_ids = tokenizer.encode(" bonjour")

datum = types.Datum(
    model_input=types.ModelInput.from_ints(prompt_ids + target_ids),
    loss_fn_inputs={
        # numeric list -> coerced to int64 TensorData automatically
        "target_tokens": target_ids,
    },
)
```

The known `loss_fn_inputs` keys and their dtypes are fixed in
`datum.py::_key_to_type`: `target_tokens` (int64), `weights` (float32),
`advantages` (float32), `logprobs` (float32), `clip_low_threshold` (float32),
`clip_high_threshold` (float32). Use `weights`/`advantages` for the RL losses
(see [SFT vs RL](../concepts/sft-vs-rl.md)); `cross_entropy` needs only
`target_tokens`.

For **robot/multimodal** inputs (images + state + action, e.g. DreamZero /
GR00T), you do not hand-build the `Datum`. You build a *collate dict* and encode
it with the DreamZero serde, exactly as the smoke does:

```python
from cybernetics.lib.dreamzero import serde

collate = build_synthetic_droid_collate(num_frames=9, rng=rng)  # your loader
datum = serde.collate_to_datum(collate)
print("loss_keys=", sorted(datum.loss_fn_inputs))  # serde fills these in
```

The collate contract (image grid geometry, the `8m+1` frame family, action
token count tied to frame count) is shown in
`build_synthetic_droid_collate` in `dreamzero_sft_smoke.py` and the
`_episode_to_collate` mapping in `finetune_on_real_data.py`. We will not
restate dataset-specific shapes here — confirm them against your data; the
template's `# TODO` comments mark every seam.

---

## One optimizer step

`forward_backward` and `optim_step` both return an `APIFuture` immediately; the
work runs on the backend. Call `.result()` (optionally with `timeout=`) to
block, or `await` the future in async code.

```python
from cybernetics import types

# 1. gradients
fb_future = training.forward_backward([datum], "cross_entropy")

# 2. apply the update (defaults shown below)
optim_future = training.optim_step(types.AdamParams(learning_rate=1e-4))

# 3. wait
fb = fb_future.result(timeout=900)
optim_future.result(timeout=900)

print(f"loss_fn_output_type={fb.loss_fn_output_type} metrics={fb.metrics}")
```

`forward_backward` returns a `types.ForwardBackwardOutput` with:

- `loss_fn_outputs: List[LossFnOutput]` — per-datum loss outputs.
- `loss_fn_output_type: str` — the record class name (e.g. `"TorchLossReturn"`).
- `metrics: Dict[str, float]` — training metrics. For MoE base models this
  includes expert-balance metrics (`e_frac_with_tokens:mean`,
  `e_max_violation:mean`, …); the docstring on `ForwardBackwardOutput` explains
  each. Watch how they *evolve*, not their initial values.

`optim_step` returns a `types.OptimStepResponse` whose only field is
`metrics: Optional[Dict[str, float]]`.

> Ordering matters. The `TrainingClient` assigns a monotonic sequence id to each
> request and the backend executes them in submission order. Submit your
> `forward_backward` for a step **before** the `optim_step` for that step. You do
> not have to block on the `forward_backward` future before submitting
> `optim_step` — the sequencing is preserved server-side — but it is clearer to
> read the loss each step.

### AdamParams defaults

`optim_step` takes a `types.AdamParams`. The optimizer is decoupled-weight-decay
Adam (equivalent to `torch.optim.AdamW`), but note Cybernetics defaults
`weight_decay` to `0.0` (no decay), unlike PyTorch. Real defaults from
`src/cybernetics/types/optim_step_request.py`:

```python
types.AdamParams(
    learning_rate=0.0001,   # 1e-4
    beta1=0.9,
    beta2=0.95,             # note: 0.95, not torch's 0.999
    eps=1e-12,
    weight_decay=0.0,       # no decay by default
    grad_clip_norm=0.0,     # 0.0 means no clipping
)
```

The field names are `beta1` / `beta2` (two separate floats), **not** a `betas`
tuple. `grad_clip_norm` clips the global gradient norm; `0.0` disables it. The
DreamZero smoke overrides these for robot SFT:

```python
adam = types.AdamParams(
    learning_rate=1e-4,
    beta1=0.95,
    beta2=0.999,
    eps=1e-8,
    weight_decay=1e-5,
    grad_clip_norm=1.0,
)
training.optim_step(adam).result(timeout=900)
```

---

## The actual loop

Neither shipped example loops — they do one step. Here is the realistic shape of
a multi-step SFT loop, composed from the same calls. Substitute your own batch
iterator for `your_batches()`.

```python
from cybernetics import types

adam = types.AdamParams(learning_rate=1e-4, weight_decay=0.0, grad_clip_norm=1.0)

for step, batch in enumerate(your_batches()):       # batch: List[types.Datum]
    fb_future = training.forward_backward(batch, "cross_entropy")
    optim_future = training.optim_step(adam)

    fb = fb_future.result(timeout=900)
    optim_future.result(timeout=900)

    loss = fb.metrics.get("loss")  # exact metric keys depend on the loss/backend
    print(f"step={step} loss={loss} metrics={fb.metrics}")

    if step % 100 == 0 and step > 0:
        ckpt = training.save_state(f"sft-step-{step}").result(timeout=900)
        print(f"checkpoint={ckpt.path}")
```

Notes that are true to the source, not aspirational:

- **`data` is a list; it is chunked for you.** Inside `forward_backward`, the
  client splits your `List[Datum]` into wire chunks (`MAX_CHUNK_LEN = 1024`
  data, `MAX_CHUNK_BYTES_COUNT = 5_000_000` bytes) and combines the results.
  One `forward_backward` call = one gradient accumulation over the whole list,
  regardless of how many chunks it became. You do not manage chunking.
- **A "batch" is whatever list you pass.** There is no `batch_size` argument; the
  number of `Datum` you put in the list per `forward_backward` is your batch.
- **The metric key for loss is not guaranteed to be `"loss"`.** `metrics` is a
  free-form `Dict[str, float]` populated by the backend for the chosen loss.
  Inspect it once (`print(fb.metrics)`) and key off what you actually see rather
  than assuming a name. This is a real gap — the SDK does not pin the schema.
- **There is no built-in epoch / shuffling / LR schedule.** If you want a
  schedule, change `learning_rate` on the `AdamParams` you pass each step.

For the async API, every method has an `_async` twin
(`forward_backward_async`, `optim_step_async`, `save_state_async`,
`load_state_async`) that you `await`; the signatures match.

---

## Checkpointing: `save_state` / `load_state`

`save_state(name, ttl_seconds=None)` persists the **model weights** and returns
an `APIFuture[SaveWeightsResponse]`. The response `.path` is a `worldlines://`
URI. `ttl_seconds=None` means the checkpoint never expires.

```python
resp = training.save_state("sft-step-1000").result(timeout=900)
print(resp.path)              # e.g. worldlines://<run-id>/weights/sft-step-1000
print(resp.has_optimizer_state)  # bool | None
```

`load_state(path)` restores **weights only** — optimizer state (Adam momentum)
is *not* restored and resets:

```python
training.load_state("worldlines://<run-id>/weights/sft-step-1000").result(timeout=900)
# continue training; optimizer momentum starts fresh
```

To also restore optimizer state for an exact resume, use
`load_state_with_optimizer(path)`. This is **backend-gated**: if the backend
does not support optimizer restore it raises `ValueError` with
`"This backend does not support optimizer-state restore. Use load_state() for
weights-only restore instead."` Fall back to `load_state` in that case.

For checkpoint lifecycle, paths, and TTLs in depth, see
[checkpoints](./checkpoints.md).

---

## Going from training to inference

When you want to sample from the model you just trained, do not re-load weights
yourself. Either save weights for a sampler and create a client from the path:

```python
saved = training.save_weights_for_sampler("sampler-001").result(timeout=900)
sampling = training.create_sampling_client(saved.path)
```

…or use the one-shot helper that does both:

```python
sampling = training.save_weights_and_get_sampling_client()  # ephemeral checkpoint
# or name it to get a persistent worldlines:// checkpoint:
sampling = training.save_weights_and_get_sampling_client(name="sft-final")
```

Then sample as described in [sampling & inference](./sampling.md).

---

## Failure modes you will actually hit

These are raised from the source you are calling, so handle them explicitly:

- **`model_id must be set before calling forward...`** — the `TrainingClient` has
  no `model_id`. You must obtain the client from
  `service_client.create_lora_training_client(base_model=...)` (or construct it
  with an existing `model_id`). A bare `TrainingClient` cannot train.
- **`This backend does not support Cybernetics training operations. Use
  create_sampling_client() ...`** — you pointed at an **inference-only** backend.
  `forward_backward` / `optim_step` / `forward` all assert training support
  first and raise `ValueError` here. Use a training-capable backend.
- **`backend does not advertise cross_entropy`** (your own check) — call
  `client.get_server_capabilities()` and confirm your loss is in
  `caps.loss_families` before submitting, as the smoke does.
- **`target_tokens must be provided when using cross_entropy`** — your `Datum`'s
  `loss_fn_inputs` is missing the `target_tokens` key.
- **`<key> must be a rectangular numeric array; ragged nested lists are not
  supported`** — you passed a ragged Python list into `loss_fn_inputs`. Pad to a
  rectangular array (or pass a numpy array of uniform shape).
- **Timeouts / cold starts.** Model creation and the first step can be slow on
  cold start. The examples pass generous `timeout=` values (e.g. `900`, `1200`
  seconds) to `.result(...)` and to `create_lora_training_client(timeout=...)`.
  Do the same; a too-small timeout raises rather than waits.

If a remote run created a session and you are running a script, cancel the
session on exit so you do not leave paid GPU running — the smokes do this in a
`finally:` via `client.create_rest_client().cancel_session(client.session_id)`.

---

## See also

- [SFT vs RL](../concepts/sft-vs-rl.md) — choosing `loss_fn` and which
  `loss_fn_inputs` each loss needs.
- [Mental model](../concepts/mental-model.md) — how clients, futures, and the
  backend fit together.
- [Sampling & inference](./sampling.md) — using the trained weights.
- [Checkpoints](./checkpoints.md) — `worldlines://` paths, TTLs, resume.
- [Client API](../reference/client-api.md) — exhaustive `TrainingClient`
  signatures.
- [Data contracts](../reference/data-contracts.md) — `Datum`, `ModelInput`,
  `TensorData`, `AdamParams` field reference.
- [Errors & troubleshooting](../reference/errors.md).
