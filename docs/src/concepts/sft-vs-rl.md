# SFT vs RL

This page explains a design decision that is easy to miss because it is enforced
by a test rather than by a type: **there is exactly one training client, and the
difference between supervised fine-tuning (SFT) and reinforcement learning (RL)
is data, not plumbing.** You do not pick a `SFTTrainer` or an `RLTrainer`. You
call the same `TrainingClient.forward_backward(...)` either way, and you select
the algorithm with two things:

1. the `loss_fn` string you pass, and
2. which keys you put in each `Datum.loss_fn_inputs`.

If you are looking for the step-by-step recipe, see the
[training guide](../guides/training.md). This page is about *why* the API is
shaped this way and *what the contract actually is*, so that you (or your coding
agent) can construct correct `Datum`s without guessing.

For where `TrainingClient` sits relative to `ServiceClient`, `SamplingClient`,
and the rest, read the [mental model](./mental-model.md) first.

## One client, selected by the loss function

The contract is asserted directly in `tests/test_one_training_client.py`. The
module docstring states it plainly:

```text
Contract: there is exactly ONE TrainingClient, and SFT vs RL is selected purely
by the loss_fn argument + the Datum.loss_fn_inputs keys -- never a second
client type or method.
```

You obtain the one client from `ServiceClient.create_lora_training_client(...)`
(see the [client API reference](../reference/client-api.md)). There is no second
factory for RL. The same instance can do SFT on one step and RL on the next:

```python
from cybernetics import types

# SFT step: cross-entropy on supervised targets
training_client.forward_backward(sft_data, "cross_entropy")

# RL step: PPO on the SAME client instance
training_client.forward_backward(rl_data, "ppo")

# DreamZero reward-weighted step, still the same client
training_client.forward_backward(rl_data, "flow_rwr")
```

`SamplingClient` is a *different role* (you sample rollouts from it), not a second
training client. The test asserts `SamplingClient is not TrainingClient` to make
that distinction explicit — generating RL rollouts and applying RL gradients are
two separate objects. See [sampling & inference](../guides/sampling.md).

### The method signature you actually call

From `src/cybernetics/lib/public_interfaces/training_client.py`:

```python
def forward_backward(
    self,
    data: List[types.Datum],
    loss_fn: types.LossFnType,
    loss_fn_config: Dict[str, float] | None = None,
) -> APIFuture[types.ForwardBackwardOutput]:
    ...
```

`forward()` (no gradients) has the same three parameters. Both return an
[`APIFuture`](../reference/client-api.md); call `.result()` (sync) or `await` it.
The `loss_fn` argument is typed `LossFnType`, which is a closed `Literal` — your
type checker will reject a misspelled loss name before you ever hit the wire.

## The loss-function vocabulary

`LossFnType` is defined in full in `src/cybernetics/types/loss_fn_type.py`:

```python
LossFnType: TypeAlias = Literal[
    "cross_entropy",
    "importance_sampling",
    "ppo",
    "cispo",
    "dro",
    "flow_rwr",
]
```

These are the only six accepted values. Honest notes on each, based on what the
source and tests actually demonstrate:

| `loss_fn` | Family | Demonstrated in tests? | Notes |
|---|---|---|---|
| `cross_entropy` | SFT | Yes | The supervised loss. Backend computes `L = sum(-logprobs * weights)` (see below). The only loss the custom-loss helpers build on. |
| `importance_sampling` | RL | Yes | Off-policy policy-gradient with importance weights. |
| `ppo` | RL | Yes | Clipped policy-gradient. |
| `flow_rwr` | RL (DreamZero) | Yes | Reward-weighted regression. This is the one divergence from the upstream Tinker loss set — it is a Worldlines/DreamZero addition, per `tests/test_import_contract.py`. |
| `cispo` | RL | No | Present in the `Literal` and accepted by the type, but no test in this repo exercises it. Treat as experimental: confirm backend support before relying on it. |
| `dro` | RL | No | Same status as `cispo` — accepted by the type, not exercised here. Treat as experimental. |

The split between the first five (the "Tinker" losses) and `flow_rwr` is asserted
in `tests/test_import_contract.py`:

```python
assert {"cross_entropy", "importance_sampling", "ppo", "cispo", "dro"} <= losses
assert "flow_rwr" in losses  # the one Worldlines/DreamZero divergence from Tinker
```

> Note on naming: **Cybernetics** is the product; **Worldlines** is the legacy
> name that survives in `worldlines://` checkpoint paths and the deprecated
> `WORLDLINES_API_KEY`; **CP** is "Cybernetic Physics" (as in `CP_API_KEY` and
> `cp_live_` keys). `flow_rwr` belongs to the DreamZero recipe, which is part of
> the Worldlines lineage. See [authentication](../guides/authentication.md) for
> the key/brand reconciliation.

## The real contract: `Datum.loss_fn_inputs` keys

A `Datum` (`src/cybernetics/types/datum.py`) has exactly two fields:

```python
class Datum(StrictBase):
    loss_fn_inputs: LossFnInputs   # Dict[str, TensorData]
    model_input: ModelInput
```

`model_input` is the token sequence the model runs on.
`loss_fn_inputs` is the per-datum dictionary of tensors the loss function
consumes. `LossFnInputs` is just `Dict[str, TensorData]`
(`src/cybernetics/types/loss_fn_inputs.py`), so the *type* permits any string
keys. The meaningful keys, and their dtypes, are fixed by the `_key_to_type` map
at the bottom of `datum.py`:

| `loss_fn_inputs` key | dtype | Used by |
|---|---|---|
| `target_tokens` | `int64` | every loss — the next-token labels |
| `weights` | `float32` | SFT (`cross_entropy`); per-token loss weights |
| `advantages` | `float32` | RL — per-token advantage estimates |
| `logprobs` | `float32` | RL — sampling-time log-probs (the behavior policy) |
| `clip_low_threshold` | `float32` | clipped RL losses — per-token lower clip bound |
| `clip_high_threshold` | `float32` | clipped RL losses — per-token upper clip bound |

```python
_key_to_type = {
    "target_tokens": "int64",
    "weights": "float32",
    "advantages": "float32",
    "logprobs": "float32",
    "clip_low_threshold": "float32",
    "clip_high_threshold": "float32",
}
```

### Two important, easy-to-miss facts about this map

**1. The map only governs dtype coercion of Python lists.** When you pass a plain
Python list as a value, `Datum`'s validator (`_maybe_convert_array`) looks the key
up in `_key_to_type` to decide whether to cast to `int64` or `float32`. A key that
is **not** in the map will raise `KeyError` if you hand it a list. If you instead
pass a fully-built `TensorData` (or a `torch.Tensor` / `numpy.ndarray`), the value
is taken as-is and any key string is accepted. So the map is a convenience for the
list path, not an exhaustive schema validator.

**2. The SDK does not enforce which keys a given `loss_fn` requires.** Nothing on
the client side checks that `ppo` was handed `advantages`, or that
`cross_entropy` was handed `weights`. The `Datum` will happily serialize whatever
keys you provide; the **backend** is what validates that the key set matches the
loss. The "required key sets" below are therefore a *convention demonstrated by
the tests and enforced server-side*, not a client-side guarantee. If you send the
wrong keys, you will find out at request time, not construction time — see
[errors & troubleshooting](../reference/errors.md).

## SFT: `cross_entropy` with `target_tokens` + `weights`

This is the supervised case. From `tests/test_one_training_client.py`:

```python
def _sft_datum() -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[1, 2, 3], dtype="int64", shape=[3]),
            "weights": types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
        },
    )
```

The test asserts that driving this datum with `"cross_entropy"` records exactly:

```python
assert holder.calls == [("cross_entropy", ["target_tokens", "weights"])]
```

**Why `weights`?** The backend cross-entropy is `L = sum(-logprobs * weights)`
(stated in a code comment in `training_client.py`). So `weights` is the per-token
multiplier on the negative log-likelihood. Set them all to `1.0` for ordinary SFT;
set prompt-token weights to `0.0` to train only on the completion (mask the
prompt); use fractional weights to down-weight tokens. This is also the hook the
custom-loss path uses: a client-side custom loss is back-propagated by sending
`weights = -dC/dlogprobs` against `cross_entropy`.

You can build `loss_fn_inputs` values three ways, all handled by the `Datum`
validator: an explicit `TensorData` (as above), a `numpy.ndarray`, or a plain
Python list (which gets cast per `_key_to_type`). Lists must be rectangular —
ragged nested lists raise `ValueError`.

## RL: a policy-gradient loss with `target_tokens` + `logprobs` + `advantages`

The RL case adds the two tensors a policy-gradient update needs: the
sampling-time `logprobs` (what the behavior policy assigned) and the per-token
`advantages`. From the same test file:

```python
def _rl_datum() -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[1, 2, 3], dtype="int64", shape=[3]),
            "logprobs": types.TensorData(data=[-0.1, -0.2, -0.3], dtype="float32", shape=[3]),
            "advantages": types.TensorData(data=[1.0, 0.5, -0.5], dtype="float32", shape=[3]),
        },
    )
```

Driving this datum through both `importance_sampling` and `ppo` records the same
key set under the same single client (note the keys come back **sorted**):

```python
assert holder.calls == [
    ("importance_sampling", ["advantages", "logprobs", "target_tokens"]),
    ("ppo", ["advantages", "logprobs", "target_tokens"]),
]
```

For PPO-style clipping you may additionally supply `clip_low_threshold` and
`clip_high_threshold` (both `float32`) per datum. They are in `_key_to_type` and
thus first-class per-token tensors. The clip bounds can *also* be passed globally
through `loss_fn_config` (next section) — choose per-token tensors when you need
token-varying bounds, the config scalar when one bound covers the batch.

### Where do `advantages` come from?

Advantages and reward-weighted-regression weights are **computed on the driver
(client) side**, not by the runtime. The SDK ships numpy ports of the DreamZero
reductions in `cybernetics/lib/dreamzero/lr_schedule.py`
(`standardized_advantages`, `group_advantages`, `rwr_weights`) so the driver can
reproduce them without importing the training stack. Per that module's docstring:

```text
the RL advantage and RWR-weight reductions run client-side and ride
loss_fn_inputs (NOT loss_fn_config, which is Dict[str, float]-only).
```

In other words: you sample rollouts with a `SamplingClient`, score them, reduce
rewards to per-token `advantages` (or `flow_rwr` weights) on the client, pack them
into `loss_fn_inputs`, and submit them to the *same* `TrainingClient`. These
helpers live under `lib/dreamzero` and are driver-side utilities — the supported,
blessed surface is still the high-level `TrainingClient` / `SamplingClient`, not
the internal transport layer.

## `loss_fn_config`: scalar hyperparameters only

`loss_fn_config` is the third argument to `forward` / `forward_backward`. Its type
is the load-bearing constraint, from `forward_backward_input.py`:

```python
loss_fn_config: Optional[Dict[str, float]] = None
"""Optional configuration parameters for the loss function (e.g., PPO clip thresholds, DPO beta)"""
```

Two things to take from this:

- It is `Dict[str, float]` — **scalars only**. Anything that varies per token or
  per sequence (advantages, weights, per-token clip bounds) belongs in
  `loss_fn_inputs` as a `TensorData`, never here. The `lr_schedule.py` docstring
  calls this out explicitly.
- The docstring's examples are illustrative, and one of them is a trap: it
  mentions **"DPO beta"**, but there is no `dpo` value in `LossFnType`. Do not
  read that comment as a promise that a DPO loss exists in this SDK — it does not.
  "PPO clip thresholds" is the realistic example; pass the clip bound as, e.g.,
  `{"clip_high_threshold": 0.2}` when you want a single batch-wide value instead
  of per-token tensors.

If a loss family needs no scalar config, leave it `None` (the default).

## What comes back

Both `forward` and `forward_backward` resolve to a
`ForwardBackwardOutput` (`src/cybernetics/types/forward_backward_output.py`):

```python
class ForwardBackwardOutput(BaseModel):
    loss_fn_output_type: str          # e.g. "TorchLossReturn", "ArrayRecord"
    loss_fn_outputs: List[LossFnOutput]   # List[Dict[str, TensorData]]
    metrics: Dict[str, float]
```

`forward_backward` accumulates gradients on the server; it does **not** update
weights. You apply the update separately with `optim_step(types.AdamParams(...))`
— the same call regardless of SFT or RL. `AdamParams` defaults
(`optim_step_request.py`) are `learning_rate=0.0001`, `beta1=0.9`, `beta2=0.95`,
`eps=1e-12`, `weight_decay=0.0`. Note the weight decay default is `0.0` (no decay),
which differs from PyTorch's `AdamW`. There is no learning-rate-schedule field on
`AdamParams`; if you want cosine/warmup, compute the per-step LR on the driver and
pass it via `learning_rate` each step (this is exactly what
`cosine_with_warmup_lr` in `lr_schedule.py` is for).

## The mental model in one paragraph

SFT and RL are the *same loop* — `forward_backward` then `optim_step` — over the
same `TrainingClient`. SFT puts `{target_tokens, weights}` in each `Datum` and
calls it with `"cross_entropy"`. RL puts `{target_tokens, logprobs, advantages}`
(plus optional clip thresholds) in each `Datum`, computes those tensors on the
driver from sampled rollouts, and calls it with `"ppo"`,
`"importance_sampling"`, or `"flow_rwr"`. The string and the keys are the only
switch. `cispo` and `dro` exist in the type but are unproven in this repo — verify
backend support before using them.

## See also

- [Mental model](./mental-model.md) — how the clients fit together
- [Training guide](../guides/training.md) — the end-to-end recipe
- [Sampling & inference](../guides/sampling.md) — generating RL rollouts
- [Data contracts](../reference/data-contracts.md) — `Datum`, `TensorData`, `ModelInput` in full
- [Client API](../reference/client-api.md) — `TrainingClient` method reference
- [Glossary](./glossary.md) — terms used above
- [Errors & troubleshooting](../reference/errors.md) — what wrong key sets look like at request time
