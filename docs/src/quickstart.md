# Quickstart

This page takes you from a clean machine to a **verifiable first training result**:
one `forward_backward` -> `optim_step` step against a real model on hosted GPU
compute, plus a fully runnable end-to-end smoke that ships in the repo.

It is a tutorial — one happy path, no branches. When you want the wider menu
(resuming from checkpoints, sampling, custom losses), follow the cross-links at
the end. For concepts behind the API, start with the
[mental model](./concepts/mental-model.md).

> **One blessed API.** Everything here uses the hand-written high-level clients —
> `ServiceClient`, `TrainingClient`, `SamplingClient`, `APIFuture` —
> re-exported at the package root (plus `RestClient`, which is hand-written too
> but reached via `service_client.create_rest_client()` rather than the package
> root). There is also a Stainless-generated
> `AsyncCybernetics` / `cybernetics.resources.*` transport layer underneath. That
> is internal plumbing; do not call it directly.

## 1. Install

The distribution name is **`cybernetic-physics`**; the import package is
**`cybernetics`**. The first training snippet below calls `get_tokenizer()`, which
needs the `transformers` dependency, so install the `[tokenizers]` extra:

```bash
pip install "cybernetic-physics[tokenizers]"
```

`numpy` is a core dependency and comes with the base install. `transformers`
(for `get_tokenizer()`), `aiohttp`, and `torch` are **not** — they live behind
extras:

| Extra | Adds | Needed for |
|---|---|---|
| `tokenizers` | `transformers` | `get_tokenizer()` |
| `aiohttp` | `aiohttp` | the aiohttp transport |
| `torch` | `torch` | local tensor interop, custom losses |
| `all` | all of the above | everything |

If you skip `[tokenizers]`, the install still succeeds, but the first call to
`get_tokenizer()` fails with `ModuleNotFoundError: No module named 'transformers'`.
See [troubleshooting](#5-if-something-breaks) below.

## 2. Authenticate

The SDK authenticates to the control plane by sending your `cp_live_` key as
`Authorization: Bearer ...`. Pick **one** of these:

```bash
# Option A — browser login (OAuth 2.0 device grant). Stores a long-lived
# cp_live_ key at ~/.config/cybernetics/auth.json.
cybernetics auth login

# Option B — export the key directly (good for CI and agents).
export CYBERNETICS_API_KEY="cp_live_..."
```

Keys are resolved in this order (first hit wins):

1. an explicit `api_key=` argument to `ServiceClient(...)`,
2. the `CYBERNETICS_API_KEY` environment variable,
3. the `CP_API_KEY` environment variable,
4. the **deprecated** `WORLDLINES_API_KEY` environment variable,
5. the stored login written by `cybernetics auth login`.

Then confirm everything is wired up. `cybernetics doctor` is a **read-only**
readiness check — it verifies API reachability, authentication, and advertised
training/sampling support **without** creating a session, model, compute lease,
or future (so it never spends GPU time):

```bash
cybernetics doctor
```

> **A note on names.** **Cybernetics** (a.k.a. **CP**, "Cybernetic Physics") is
> the product. **Worldlines** is the legacy name that survives in two places you
> will see in this SDK: checkpoint paths (`worldlines://run-id/weights/...`) and
> the deprecated `WORLDLINES_API_KEY`. Keys are prefixed `cp_live_` and the env
> var you want is `CYBERNETICS_API_KEY`. See
> [Authentication](./guides/authentication.md) for the full story.

## 3. Run your first training step

This is the minimal supported loop: create a `ServiceClient`, ask it for a LoRA
`TrainingClient`, build one `Datum`, then `forward_backward(...)` ->
`optim_step(...)`. Both training calls return an `APIFuture`; calling `.result()`
blocks until the backend finishes the work.

Save this as `first_step.py`:

```python
import cybernetics
from cybernetics import types

# Near-instant: opens a Worldlines session backed by your credentials.
service_client = cybernetics.ServiceClient(project_id="quickstart")

# Takes a moment: allocates the model and a compute lease, then cold-starts it.
# `timeout` bounds the cold-start wait. Defaults: rank=32, train_mlp/attn/unembed=True.
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B",
    rank=16,
    timeout=900,
)

# `get_tokenizer()` is why we installed the [tokenizers] extra.
tokenizer = training_client.get_tokenizer()

# One illustrative training example. `loss_fn_inputs` carries the cross-entropy
# targets as TensorData; `weights` lets you up/down-weight per-token loss.
prompt_tokens = tokenizer.encode("The capital of France is")
target_tokens = tokenizer.encode(" Paris")
datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=prompt_tokens),
    loss_fn_inputs={
        "target_tokens": types.TensorData(
            data=target_tokens,
            dtype="int64",
            shape=[len(target_tokens)],
        ),
        "weights": types.TensorData(
            data=[1.0] * len(target_tokens),
            dtype="float32",
            shape=[len(target_tokens)],
        ),
    },
)

# Compute gradients, then take one Adam step. `.result()` blocks on each.
fb = training_client.forward_backward([datum], "cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()

print("metrics:", fb.metrics)
print("loss_fn_output_type:", fb.loss_fn_output_type)
print("num loss_fn_outputs:", len(fb.loss_fn_outputs))
```

Run it:

```bash
python first_step.py
```

### What success looks like

`forward_backward(...).result()` returns a `ForwardBackwardOutput`. The fields you
get back (copied from the type) are:

- `metrics: dict[str, float]` — training metrics. Keys are formatted
  `name:reduction` (for example a loss metric reduced as `:mean`). This is where
  you watch the loss move.
- `loss_fn_outputs: list[dict[str, TensorData]]` — one record per `Datum`
  (here, one).
- `loss_fn_output_type: str` — the record class name, e.g. `"TorchLossReturn"`.

`optim_step(...).result()` returns an `OptimStepResponse` and applies the update.

So a successful run prints something like:

```text
metrics: {'loss:mean': 7.42, ...}
loss_fn_output_type: TorchLossReturn
num loss_fn_outputs: 1
```

The exact metric keys and values depend on the model and your data — **do not
hardcode them**. The signal that the loop works is simple: **no exception is
raised, `.result()` returns, and `metrics` is populated.** Run a few steps in a
loop and the loss-style metric should trend **down**. (For a concrete,
documented loss curve on a shipped example, see the next section.)

> `Qwen/Qwen3-8B` is the canonical example model used throughout the client API.
> If your backend does not serve it you will get an error at
> `create_lora_training_client`; check what is available with
> `service_client.get_server_capabilities().supported_models`.

## 4. Verify end-to-end with the shipped smoke

The hand-written loop above proves the surface. For a **fully runnable,
end-to-end** result with documented numbers, the repo ships
`examples/dreamzero_sft_smoke.py` — the SDK-native DreamZero LoRA-SFT template
that all other examples mirror.

It is **local-only by default**: with no flags it builds the synthetic batch,
encodes one `Datum`, prints the wire keys, and exits — **no session, no GPU, no
spend**:

```bash
python examples/dreamzero_sft_smoke.py
```

Expected local output (truncated):

```text
built_datum=true base_model=dreamzero-droid loss_keys=... num_frames=9
sample_loss_keys=...
remote_run=false
```

To actually run `forward_backward` -> `optim_step` -> `save_state` on hosted GPU,
check readiness first, then add `--remote-run`. A cold backend image can take
several minutes to pull and load weights, so give it a generous `--timeout`:

```bash
cybernetics doctor
python examples/dreamzero_sft_smoke.py --remote-run --timeout 2400
```

On a successful remote run the script prints lines like:

```text
forward_backward_done=true model_id=... result=...
checkpoint=...
cleanup_session_cancelled=true session_id=... stopped_lease_ids=...
```

The validated `dreamzero-droid` run in the example library shows the loss moving
**6.08 -> 0.44** over a short run. Treat that as a smoke-level signal on synthetic
fixtures that gradients flow the right direction — **not** a benchmark or a
converged policy.

> **The run cancels its own session on exit by default**, so a successful smoke
> does not leave paid compute running. Pass `--keep-lease` only when you
> deliberately want to debug inside the (billed) container.

## 5. If something breaks

| Symptom | What it means | Do this |
|---|---|---|
| `cybernetics.AuthenticationError` (HTTP 401) | No valid key reached the control plane. | Run `cybernetics auth login`, or `export CYBERNETICS_API_KEY="cp_live_..."`, then re-run `cybernetics doctor`. |
| `ModuleNotFoundError: No module named 'transformers'` | The `[tokenizers]` extra is not installed, but `get_tokenizer()` needs it. | `pip install "cybernetic-physics[tokenizers]"`. |
| `ModuleNotFoundError: No module named 'cybernetics'` | You installed but imported the wrong name. | Install is `cybernetic-physics`; the **import** is `import cybernetics`. |
| `ValueError: This backend does not support Cybernetics training operations.` | You aimed a `TrainingClient` call at an inference-only backend. | Use a training backend, or switch to sampling — see [Sampling & Inference](./guides/sampling.md). |
| `ValueError: Either model_path or base_model must be provided` | You called `create_sampling_client()` with neither argument. | Pass `base_model=...` or a checkpoint `model_path=...`. |
| Hangs on `create_lora_training_client` | Cold start: the backend is pulling/loading the model image. | This is expected on a cold backend. Increase `timeout=` and watch the queue logs the SDK prints. |

For the full error taxonomy and exit codes, see
[Errors & Troubleshooting](./reference/errors.md).

## Where to go next

- [Mental Model](./concepts/mental-model.md) — sessions, leases, futures, and how
  the clients fit together.
- [SFT vs RL](./concepts/sft-vs-rl.md) — which loss path you are actually on.
- [Training](./guides/training.md) — real datasets, batching, and the full
  training loop.
- [Sampling & Inference](./guides/sampling.md) — turn a trained checkpoint into a
  sampler.
- [Checkpoints](./guides/checkpoints.md) — `save_state`, `worldlines://` paths,
  and resuming.
- [Client API](./reference/client-api.md) — exhaustive signatures for every
  client method used here.
- [CLI](./reference/cli.md) — `auth`, `doctor`, and the rest of the commands.
