# Finetuning robot foundation models with Worldlines

This directory is a worked example library for finetuning robot foundation
models on the hosted Worldlines / Cybernetics control plane through the
`cybernetics` Python SDK. Each backend speaks the SAME Tinker-style contract:

```
forward_backward([datum], loss_fn)  ->  optim_step(adam)  ->  save_state(name)
```

PI0 DROID serving is deliberately outside that training contract. The
inference-only `pi0-droid` model accepts a typed `DroidObservation` and returns
one `[H, 8]` chunk of absolute DROID joint-position/gripper targets. It does not
support SDE, predicted video, LoRA/full training, `forward_backward`, or
`optim_step`.

The CLIENT builds a per-model `collate` dict locally (tokenization, image stacks,
normalized actions), encodes it into ONE `Datum` per sample with that model's
serde, and ships it over the wire. The runtime decodes the `Datum` back into the
model's native input and runs the real forward/backward. No model dataset is
instantiated server-side, and the serdes are pure `numpy`/`torch` (no GPU, no
`groot`/`openpi`/`diffusers` needed to build a datum).

## The four validated backends

All four were validated end-to-end on a real H200 (datum -> `forward_backward` ->
`optim_step` -> `save_state`):

| `base_model`      | What it is                                              | Train mode             | Loss (server-side)                  |
| ----------------- | ------------------------------------------------------- | ---------------------- | ----------------------------------- |
| `dreamzero-droid` | DreamZero flow-matching VLA (Wan-lineage), DROID recipe | LoRA                   | timestep-reweighted velocity MSE    |
| `groot-n1.5`      | GR00T N1.5 — the Wan-lineage groot; SAME datum as above | LoRA                   | timestep-reweighted velocity MSE    |
| `pi0.5`           | openpi pi0.5 vision-language-action model               | full finetune          | flow-matching velocity MSE          |
| `cosmos3-nano`    | NVIDIA Cosmos 3 omni world model (video)                | LoRA                   | rectified-flow velocity MSE         |

`groot-n1.5` consumes the EXACT same `dreamzero_cotrain.collate` datum as
`dreamzero-droid` — they share `cybernetics.lib.dreamzero.serde`; only the
`base_model` string differs on the wire.

For `pi0.5` and `cosmos3-nano` the `loss_fn` string passed to `forward_backward`
is IGNORED server-side (the loss is the model-internal flow / rectified-flow MSE);
the examples ship the wire literal `"cross_entropy"` because the wire
`LossFnType` Literal has no flow-matching name.

### Validated results

These are the numbers observed on the H200 validation runs (short runs; they show
the loss/gradient moving in the right direction, not a converged policy):

| Backend           | Observed                                   |
| ----------------- | ------------------------------------------ |
| `dreamzero-droid` | loss 6.08 -> 0.44                          |
| `groot-n1.5`      | loss 1.84 -> 0.24                          |
| `pi0.5`           | grad norm 6.3 -> 5.0                       |
| `cosmos3-nano`    | loss 0.35 -> 0.32                          |

Treat these as smoke-level signals on synthetic fixtures, not benchmark claims.

## Examples

| Script                       | Backend           | Notes                                                       |
| ---------------------------- | ----------------- | ----------------------------------------------------------- |
| `pi0_droid_sampling.py`      | `pi0-droid`       | authenticated typed DROID inference; no training            |
| `dreamzero_sft_smoke.py`     | `dreamzero-droid` | LoRA-SFT smoke (the template all others mirror)             |
| `dreamzero_rl_smoke.py`      | `dreamzero-droid` | flow-RWR / PPO RL smoke (world-model + trajectory tensors)  |
| `groot_sft_smoke.py`         | `groot-n1.5`      | reuses the dreamzero serde + synthetic DROID collate        |
| `pi05_sft_smoke.py`          | `pi0.5`           | full-finetune; `cybernetics.lib.pi05`                       |
| `cosmos_sft_smoke.py`        | `cosmos3-nano`    | rectified-flow LoRA; `cybernetics.lib.cosmos`               |
| `flywheel_demo.py`           | `cosmos3-nano` + `groot-n1.5` | the synthetic-data flywheel slice               |
| `finetune_on_real_data.py`   | `groot-n1.5`      | TEMPLATE: finetune on a real LeRobot dataset                |
| `push_checkpoint_to_hf.py`   | —                 | upload a saved checkpoint to a HuggingFace repo             |

## PI0 DROID sampling

Capture one DROID observation as an NPZ file with keys
`exterior_image_0_left`, `exterior_image_1_left`, `wrist_image_left`,
`joint_position`, and `gripper_position`. The images must be `H x W x 3` uint8
RGB arrays; the state contains seven joints and one gripper value.

```bash
cybernetics auth login
python pi0_droid_sampling.py observation.npz \
  --instruction "pick up the cube" \
  --output pi0-actions.npy
```

Validate a captured observation without authentication, network work, or GPU
allocation:

```bash
python pi0_droid_sampling.py observation.npz \
  --instruction "pick up the cube" \
  --validate-only
```

## Local-vs-remote pattern

Every smoke is **local-only by default**. With no flag it builds the collate,
encodes the `Datum`, and prints the wire keys — no session, no GPU, no spend:

```bash
python dreamzero_sft_smoke.py
python groot_sft_smoke.py
python pi05_sft_smoke.py
python cosmos_sft_smoke.py
python flywheel_demo.py
```

Pass `--remote-run` to create a Worldlines session/model and spend GPU time on
the configured control plane. Remote runs **cancel their SDK session on exit** so
a successful smoke does not leave paid compute running; pass `--keep-lease` to
keep the lease alive for debugging.

```bash
python groot_sft_smoke.py --remote-run            # cancels the session on exit
python groot_sft_smoke.py --remote-run --keep-lease   # leaves compute running
```

## Pointing at a backend

```bash
export CYBERNETICS_BASE_URL="https://your-control-plane"   # or pass --base-url
export CYBERNETICS_API_KEY="cp_live_..."                   # or WORLDLINES_API_KEY, or --api-key
```

The examples construct the client with the API key forwarded as the
`X-API-Key` header (matching the validated drivers):

```python
client = cybernetics.ServiceClient(
    base_url=args.base_url,
    api_key=api_key,
    default_headers={"X-API-Key": api_key},
    user_metadata={"example": "..."},
)
```

`--base-url` defaults to `CYBERNETICS_BASE_URL` (then `CP_API_BASE`, then your
stored login). The API key resolves from `--api-key`, then `CYBERNETICS_API_KEY`,
then the deprecated `WORLDLINES_API_KEY`.

## The flywheel

`flywheel_demo.py` wires one slice of the synthetic-data flywheel:

```
Cosmos3-Nano dream-video
    -> placeholder-IDM pseudo-actions
    -> a dreamzero/groot collate (images = the dreamed frames)
    -> forward_backward on groot-n1.5
```

A world model dreams robot-manipulation video, an inverse-dynamics model (IDM)
labels the dreamed frames with pseudo-actions, and the VLA trains on
`(dreamed video, pseudo-actions)` — generated data improving the policy with no
new teleop.

The IDM in this demo is a **placeholder** (a crude frame-diff motion proxy). The
real IDM is NVIDIA GR00T-Dreams `idm_training.py`. In local mode the demo
synthesizes stand-in frames so it runs without a GPU; pass `--cosmos-checkpoint`
(needs the cosmos venv) to dream a real clip, or `--dream-npy` to load one.

## On-H200 venvs

The serdes are pure `numpy`/`torch` and build a datum anywhere. But the actual
model runtimes on the H200 need their own venvs:

- **cosmos** — `diffusers` + `diffusers_cosmos3` (for `cosmos3-nano` and for the
  `flywheel_demo.py --cosmos-checkpoint` generation path).
- **openpi** — the openpi stack (for `pi0.5`).

You do not need these venvs to run the local-only smokes; you only need them on
the backend that serves the corresponding `base_model` (and locally if you dream
a real Cosmos clip in the flywheel).

## Exporting a checkpoint

`save_state(name)` writes the artifact to the backend's on-disk `artifact_root`
as `/data/<run>/weights/<name>/` (`model.safetensors` + `config.json` +
`metadata.pt` + a `COMPLETE` sentinel). To publish it:

```bash
export HF_TOKEN=hf_...   # a HuggingFace WRITE token; never hardcoded
python push_checkpoint_to_hf.py \
    --checkpoint-dir /data/<run>/weights/<name> \
    --repo-id your-org/your-model \
    --private
```
