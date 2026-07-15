# Mental Model

This page is the architecture map you (or your coding agent) need before writing
real code against the SDK. It explains how the pieces fit together and *why* the
API is shaped the way it is. It is not a tutorial — for a working end-to-end
example see the [quickstart](../quickstart.md), and for exhaustive signatures see
the [client API reference](../reference/client-api.md).

Everything below describes the **blessed public surface**: the hand-written
clients in `cybernetics.lib.public_interfaces`, re-exported at the package root.
There is a second, lower layer (the generated transport) that you should *not*
call directly; it is covered at the end so you know what to avoid.

## Names you will see (read this first)

The product went through a rename, and the seams are still visible in the code.
Knowing the mapping up front saves confusion:

- **Install vs. import.** You install the distribution `cybernetic-physics` but
  you import the package `cybernetics`:

  ```bash
  pip install cybernetic-physics
  ```

  ```python
  import cybernetics
  from cybernetics import ServiceClient
  ```

- **Cybernetics** is the current product/SDK name.
- **Worldlines** is the legacy name. It survives in three places you will
  actually touch: checkpoint paths look like `worldlines://run-id/weights/checkpoint-001`,
  `ServiceClient.session_id` is documented as the "Worldlines session ID", and the
  deprecated `WORLDLINES_API_KEY` environment variable is still read for one
  release (it emits a `DeprecationWarning`).
- **CP** stands for **Cybernetic Physics**. It shows up in the `CP_API_KEY`
  environment variable and in live API keys, which are prefixed `cp_live_`.

API-key resolution order (from `cybernetics.lib.credentials.resolve_api_key`) is:
explicit `api_key=` argument → `CYBERNETICS_API_KEY` → `CP_API_KEY` →
deprecated `WORLDLINES_API_KEY` → the stored login file written by
`cybernetics auth login`. See [authentication](../guides/authentication.md) for
the full story.

## The shape of the SDK

```text
                  ┌────────────────────────────────────────┐
                  │              ServiceClient             │
                  │   entry point — owns ONE session       │
                  └────────────────────────────────────────┘
                        │  create_*()  (factory methods)
        ┌───────────────┼─────────────────────────────┐
        ▼               ▼                             ▼
  TrainingClient   SamplingClient               RestClient
  forward_backward sample()                     list/get/publish
  optim_step       compute_logprobs()           checkpoints,
  save_state /                                   training runs,
  load_state                                     sessions
        │               │                             │
        └───────────────┴─────────────────────────────┘
                        │  all three share the SAME holder + session
                        ▼
            InternalClientHolder  (session_id, async event loop,
            retries, connection pools)
                        │
                        ▼  generated AsyncCybernetics transport (internal)
                        ▼
                Cybernetics backend  (compute leases / GPUs)
```

The whole API is reachable from one object. You construct a `ServiceClient`, and
it hands you the other three clients.

## ServiceClient: the entry point and factory

`ServiceClient` is the only thing you construct directly. Constructing it is
near-instant and does two things: it resolves your credentials into request
headers, and it opens a **session** on the backend.

```python
from cybernetics import ServiceClient

# Near-instant: resolves credentials, opens a session.
client = ServiceClient()
print(client.session_id)  # the Worldlines session ID backing this client
```

From there it is a factory for the three working clients:

| Method | Returns | Cost |
| --- | --- | --- |
| `create_lora_training_client(base_model=..., rank=32, ...)` | `TrainingClient` | Takes a moment — initializes the model and assigns compute |
| `create_sampling_client(base_model=...)` / `create_sampling_client(model_path=...)` | `SamplingClient` | Near-instant |
| `create_rest_client()` | `RestClient` | Near-instant (no network call) |

There are also convenience constructors that resume from saved weights:
`create_training_client_from_state(path)` (weights only) and
`create_training_client_from_state_with_optimizer(path)` (weights **and** Adam
optimizer state). The optimizer variant raises `ValueError` with a clear message
if the backend does not support optimizer restore or the checkpoint has no
optimizer state — it does not silently fall back. See
[checkpoints](../guides/checkpoints.md).

Note the cost column matters: `create_lora_training_client` may cold-start a
model, so it accepts an optional `timeout: float | None` argument. The factory
methods that block on a cold start (`create_lora_training_client`,
`create_sampling_client`) return the client object **directly**, not a future —
they wait internally.

## The future-then-`.result()` pattern

Once you have a `TrainingClient`, the per-step training methods do **not** block.
They return immediately with a future, so you can pipeline work (submit the next
request while the previous one is still in flight) and only block when you need
the value:

```python
# Submit forward/backward and the optimizer step back to back...
fwdbwd_future = training_client.forward_backward(data, "cross_entropy")
optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))

# ...then block on the results when you actually need them.
fwdbwd_result = fwdbwd_future.result()   # blocks until gradients are computed
optim_result = optim_future.result()
print(f"loss = {fwdbwd_result.loss}")
```

The object these methods return is an `APIFuture[T]` (defined in
`cybernetics.lib.public_interfaces.api_future`). It is deliberately usable from
both sync and async code:

```python
class APIFuture(ABC, Generic[T]):
    def result(self, timeout: float | None = None) -> T: ...
    async def result_async(self, timeout: float | None = None) -> T: ...
    def __await__(self): ...   # `await future` == `await future.result_async()`
```

So all three of these are valid ways to get the value out:

```python
result = future.result()              # sync, block forever
result = future.result(timeout=30)    # sync, raise TimeoutError after 30s
result = await future                 # async, via __await__
```

`TrainingClient.forward`, `forward_backward`, `optim_step`, `save_state`,
`load_state`, `load_state_with_optimizer`, and `save_weights_for_sampler` all
return `APIFuture[...]`. The [training guide](../guides/training.md) walks
through a real loop built on this pattern.

### Two honest wrinkles in the return types

The future-then-`.result()` story is not perfectly uniform across all clients.
Two cases are worth knowing so you are not surprised:

1. **`SamplingClient.sample()` and `compute_logprobs()` return a plain
   `concurrent.futures.Future`, not an `APIFuture`.** That means `.result()`
   works but `await sampling_client.sample(...)` does **not**. For async code,
   use the explicit twin `sample_async()` / `compute_logprobs_async()` instead:

   ```python
   # Sync:
   future = sampling_client.sample(prompt, num_samples=1, sampling_params=params)
   response = future.result()

   # Async — do NOT `await` sample(); call the _async twin:
   response = await sampling_client.sample_async(prompt, 1, params)
   ```

2. **`RestClient` is mixed.** Most of its read/write methods (`get_training_run`,
   `list_checkpoints`, `list_sessions`, ...) return a `concurrent.futures.Future`
   via `.result()`. A couple — `get_weights_info_by_worldlines_path` and
   `get_sampler` — return an awaitable `APIFuture`. When in doubt, `.result()`
   works on all of them; reach for `await` only where the
   [client API reference](../reference/client-api.md) shows an `APIFuture`
   return type.

By contrast, the `ServiceClient` query methods (`get_server_capabilities`,
`get_scheduler_state`) and the `create_*` factories return the **resolved value
or object directly** — they call `.result()` for you internally.

## Sync-first design, with `_async` twins

The SDK is written sync-first: the method you reach for by default is the
blocking one, and almost every public method has an explicit async twin with the
same name plus an `_async` suffix.

```python
# Sync (the default):
caps = service_client.get_server_capabilities()

# Async twin:
caps = await service_client.get_server_capabilities_async()
```

This pairing is everywhere: `forward_backward` / `forward_backward_async`,
`create_lora_training_client` / `create_lora_training_client_async`,
`sample` / `sample_async`, and so on.

Some sync methods are decorated with `@sync_only`. This is a guardrail against a
common footgun — calling a blocking method from inside a running event loop,
which can deadlock the SDK's internal loop. Be precise about what it does, from
`cybernetics.lib.sync_only`:

- If you call a `@sync_only` method **from an async context** (a running event
  loop) and you are **not** in a Jupyter notebook, it logs a warning telling you
  to use the `_async` twin, plus a stack trace — and then **runs anyway**. It
  does not raise, and it does not block the call.
- In Jupyter (where a loop is always running) the check is suppressed, so the
  warning does not fire.

The takeaway: in plain sync scripts, use the blocking methods freely. Inside
`async def` code, prefer the `_async` twins — not because the sync ones are
forbidden, but because mixing blocking calls into an event loop is what the
warning is trying to save you from.

## The two-layer design: which layer to use

There are two layers in this package, and only one of them is the supported API.

**Layer 1 — hand-written high-level clients (use these).**
`ServiceClient`, `TrainingClient`, `SamplingClient`, `RestClient`, and
`APIFuture` live in `cybernetics/lib/public_interfaces/` and are re-exported at
the package root:

```python
from cybernetics import ServiceClient, TrainingClient, SamplingClient, APIFuture
```

These give you the session lifecycle, the future-then-`.result()` ergonomics,
the sync/async twins, retries, chunking of large requests, and queue-state
logging. This is the entire supported public API.

**Layer 2 — the generated transport (do not use directly).**
Under the hood, the high-level clients talk to a Stainless-generated async
transport. You can see it in the package root: `cybernetics.__init__` sets
`Cybernetics = AsyncCybernetics`, and the resources live under
`cybernetics.resources.*` (e.g. `client.training.forward`,
`client.models.create`, `client.weights.save`). The high-level clients reach it
through an internal `InternalClientHolder` that owns the event loop, connection
pools, and retry logic.

You will see references to `AsyncCybernetics` and `cybernetics.resources` if you
read the source or tracebacks. **Do not build against them directly.** They are
the internal wire format: the request/response shapes, sequence-id bookkeeping,
and retry semantics are managed for you by the high-level clients, and calling
the transport yourself bypasses all of that. Treat anything outside
`public_interfaces` (and the re-exported names above) as private.

## What a session and a compute lease are

These are backend concepts; the SDK only exposes them at the edges, so this
description stays at the level the source actually supports — no more.

- A **session** is the server-side context a `ServiceClient` opens when you
  construct it. It is identified by `ServiceClient.session_id` (the "Worldlines
  session ID"). Every `TrainingClient`, `SamplingClient`, and `RestClient` you
  create from that `ServiceClient` shares the same session: they are all backed
  by the same `InternalClientHolder`, which carries the session id. The session
  is what ties your training runs and samplers together server-side — you can
  look one up with `rest_client.get_session(session_id)`, which returns the
  session's `training_run_ids` and `sampler_ids`.

- A **compute lease** is the backend's claim on actual compute (GPUs) on behalf
  of a session. The SDK does not let you allocate or inspect leases directly; the
  one place the concept surfaces is `RestClient.cancel_session(session_id)`,
  whose docstring describes the lifecycle precisely: cancelling a session "marks
  session futures terminal and asks the autoscaler to stop session-owned leases
  only when no other live session or active future still references them." In
  other words, leases are reference-counted and reclaimed by an autoscaler, not
  by you.

What this means in practice: creating a training client is the expensive step
(it assigns compute to your session), and the backend — not your process — owns
the lifecycle of the underlying hardware. Beyond `cancel_session`, the SDK gives
you no further knobs over leases, and this page will not invent any.

## Where to go next

- [Quickstart](../quickstart.md) — the shortest path to a real result.
- [Training](../guides/training.md) — building a loop on the future pattern.
- [Sampling & Inference](../guides/sampling.md) — using `SamplingClient`.
- [Checkpoints](../guides/checkpoints.md) — `save_state` / `load_state` and `worldlines://` paths.
- [Client API reference](../reference/client-api.md) — exhaustive signatures and return types.
- [Glossary](./glossary.md) — quick definitions of session, lease, LoRA, and the rest.
