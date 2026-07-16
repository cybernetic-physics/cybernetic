# Cybernetics Python SDK

`cybernetic-physics` is the Python client for the hosted Cybernetics platform —
a Tinker-like API for rollout, sampling, and LoRA training on platform-managed
GPU compute leases, plus robotics and simulation-asset helpers for hosted robot
workflows.

```bash
pip install cybernetic-physics      # distribution name; the import package is `cybernetics`
export CYBERNETICS_API_KEY="cp_live_..."   # or run: cybernetics auth login
cybernetics doctor                  # read-only API/auth/SFT/RL readiness check
```

Install directly from GitHub when you want the current repo version:

```bash
uv add "cybernetic-physics @ git+https://github.com/cybernetic-physics/cybernetic.git"
pip install "cybernetic-physics @ git+https://github.com/cybernetic-physics/cybernetic.git"
```

For `pyproject.toml`:

```toml
dependencies = [
  "cybernetic-physics @ git+https://github.com/cybernetic-physics/cybernetic.git",
]
```

```python
import cybernetics
from cybernetics import types

service_client = cybernetics.ServiceClient(project_id="robotics-lab")
training_client = service_client.create_lora_training_client(base_model="dreamzero-droid")

tokenizer = training_client.get_tokenizer()  # requires the [tokenizers] extra
datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens=tokenizer.encode("example")),
    loss_fn_inputs={
        "target_tokens": types.TensorData(data=[1, 2, 3], dtype="int64", shape=[3]),
        "weights": types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
    },
)
training_client.forward_backward([datum], "cross_entropy").result()
training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()
```

## Authentication

The SDK authenticates to the control plane with your `cp_live_` key sent as
`Authorization: Bearer ...`. Keys are resolved in this order:

1. an explicit `api_key=` argument,
2. the `CYBERNETICS_API_KEY` environment variable,
3. the stored login written by `cybernetics auth login`
   (`~/.config/cybernetics/auth.json`).

## Readiness checks

Run `cybernetics doctor` before launching GPU work. It checks API reachability,
authentication, advertised training/sampling support, and DreamZero SFT/RL
capabilities without creating a Worldlines session, model, compute lease, or
future.

```bash
CYBERNETICS_BASE_URL=https://luc-api.cyberneticphysics.com cybernetics doctor
cybernetics doctor --require-rl
cybernetics --format json doctor
```

`--require-rl` exits nonzero unless the backend advertises the requested
DreamZero RL loss family (`flow_rwr` by default). This keeps SFT-only
deployments usable while making RL readiness explicit in CI and runbooks.

## Session replay for agents

`cybernetics.replay` reads the durable session timeline and produces bounded,
provider-neutral observations for text-and-image agents. Exact nanosecond values
remain Python integers in memory and decimal strings in JSON. The default query
is the latest 30 seconds, capped at 100 events and four visual frames. Timestamp
values must fit the non-negative PostgreSQL `BIGINT` range. Dense observations
retain the newest matching suffix and return it in chronological order; quality
metadata identifies older evidence omitted by response, candidate, byte, or
archive-scan limits. Fully implicit latest windows are anchored by the
observation endpoint itself, avoiding a summary/read race with active sessions;
custom latest durations use that server anchor and then request an exact window.

```python
import cybernetics
from cybernetics.replay import ReplayQuery

with cybernetics.Client() as client:
    summary = client.replay.describe("sess_...")
    print(summary.channels)

    raw = client.replay.select_events(
        "sess_...",
        ReplayQuery(channels=("robot/*",), max_events=100),
    )
    print(raw.matched_events, len(raw.events), raw.truncation_reason)

    observation = client.replay.get_observation(
        "sess_...",
        ReplayQuery(channels=("camera/*", "robot/*")),
        image_data=True,
    )

    # Plain dictionaries for the two common multimodal message formats. Each
    # frame is preceded by exact time, window offset, optional simulation time,
    # source, channel, and event labels; untrusted replay context is last.
    openai_content = observation.to_openai_content()
    anthropic_content = observation.to_anthropic_content()

    bundle = client.replay.export_agent_bundle("sess_...", "./replay-context")
    print(bundle.manifest_path)
```

The provider-neutral `ReplayObservation` is the SDK contract; the adapter
methods do not import either provider SDK. It validates the source
`cybernetic.replay-observation/v1` envelope and serializes a distinct
`cybernetic-replay-agent-observation/v1` shape, so wire and agent artifacts are
never mislabeled. A replay window is explicitly untrusted delta evidence, never
a claim that the full world state was reconstructed. Event `semantics` distinguish
deltas, samples, events, and predictions; `media_ids` preserve authoritative
event-to-frame linkage. Quality metadata remains additive and carries server
truncation, scan, invalid-image, and missing unit/frame warnings.
`page.available_exact` mirrors `quality.matchedEventsExact`; when false, the
available count is only a lower bound and agent artifacts state that explicitly.

The agent bundle is deterministic and safe to inspect or attach selectively:

- `manifest.json` records the exact query, bounds, hashes, truncation, warnings,
  omissions, and file inventory.
- `context.md` explains the interpretation contract and links frames by media ID
  to exact event time, window offset, simulation time, source, and channel.
- `observations.ndjson` and `events.ndjson` contain text-safe structured data.
- `frames/` contains magic-validated JPEG, PNG, WebP, or GIF bytes separately;
  textual artifacts recursively omit embedded binary fields and every data URL
  (including non-base64 SVG/text variants), and redact common
  structured, quoted-JSON, Authorization/Cookie header, vendor environment, and
  CLI-style credential values.

The CLI exposes the same boundary:

```bash
cybernetics replay inspect sess_...
cybernetics replay events sess_... --channel 'camera/*' --max-events 100
cybernetics replay events sess_... --ndjson --start-time-ns 100 --end-time-ns 200
cybernetics replay export sess_... --out ./replay-context --max-images 4
cybernetics --format json replay inspect sess_...
```

Raw `iter_events()` reads are bounded across recordings by index-page, chunk,
compressed-byte, decoded-byte, and parsed-event budgets. They validate immutable
chunk size/SHA, vendor NDJSON MIME type, gzip encoding, event counts, channel
envelopes, time envelopes, v1 recording identity, strict finite JSON, and nesting
depth. Use `select_events()` when completeness matters: its
`ReplayEventSelection` reports matched/returned counts and whether `max_events`
omitted older (`max_events_before`) or newer (`max_events_after`) evidence. JSON
CLI output includes the same page metadata, while `--ndjson` prints an explicit
stderr warning if it truncates. Narrow the time/channel/recording selection when
a session exceeds those guardrails. `recording_ids` and
`include_control_events=False` are raw-only filters; observation/export rejects
them instead of silently ignoring them. The default event serializer and every
CLI mode omit embedded image base64; callers must explicitly use the in-memory
event payload when they truly need raw bytes. Chunk pages use server-side time
overlap filters. The control-event endpoint remains cursor-only, so very long
sessions can hit its raw 64-page/10,000-event guardrail even for a narrow latest
window; use the bounded observation API or `include_control_events=False` in
that case.

## Simulation assets

The SDK includes an MVP `sim` namespace for packaging local simulation assets as
Cybernetics environments, launching hosted Isaac preview sessions, and producing
asset references that can be reused by RobotTask specs.

```bash
cybernetics sim inspect ./scene-folder --format json
cybernetics sim import ./scene.usdz --name warehouse-demo
cybernetics sim import ./robot.urdf --bundle-path robot.bundle.zip --source-url https://example.test/robot
cybernetics sim launch cybernetics://envs/env_.../versions/ver_... --wait
cybernetics sim render ./scene-folder --root-stage scene.usd --wait --out preview.jpg
```

The top-level `cybernetics.Client` is a composition root. Product namespaces
attach under it, so simulation helpers live at `client.sim` without making
`cybernetics.sim` own the package root.

```python
import cybernetics

with cybernetics.Client() as client:
    imported = client.sim.import_asset("./scene-folder", name="warehouse-demo")
    sim_asset_ref = imported.to_asset_ref().to_dict()

    preview = client.sim.render(imported, wait=True, out="preview.jpg")
    print(sim_asset_ref["uri"])
    print(preview.preview_url)
    print(preview.launch_url)
```

An authenticated SDK client can also control the hosted Isaac session through a
private, session-scoped MCP grant. The grant is pinned to one session, permits
only `isaac.*` tools, and is revoked when the MCP context closes. The local
credential is erased and disabled before remote revocation; a failed revocation
remains retryable through `SimulationClient.close()` until the control plane
confirms it or the scoped grant expires. Closing the MCP context does not stop
the Isaac session.

```python
import cybernetics

environment = "cybernetics://envs/env_.../versions/ver_..."

with cybernetics.Client() as client:
    launched = client.sim.launch(environment, wait=True)
    try:
        with client.sim.mcp_session(launched.session_id) as isaac:
            scene = isaac.call_tool("isaac.get_scene_info")
            isaac.call_tool("isaac.step_simulation", {"steps": 1})
            print(scene)
    finally:
        client.sim.stop_session(launched.session_id)
```

This MCP boundary controls simulation only. DreamZero sampling and LoRA training
continue through the Worldlines clients. A robotics loop captures RGB and
proprioception from hosted Isaac, calls `sample_droid()` (or the lower-level
continuous-policy sampling contract), and applies the returned action chunk to
the same hosted session.

Session readiness and cleanup tolerate brief control-plane gateway outages
without hiding persistent failures. `wait_for_session()` and `stop_session()`
make at most six attempts for HTTP 502, 503, or 504, using 1, 2, 4, 8, and 10
second backoffs (25 seconds total). Other HTTP errors are not retried. If a stop
receives HTTP 409, the SDK reads the session and treats the call as successful
only when the stop was already accepted (`stopping`) or the session is ended;
unrelated conflicts still raise `SimulationError`.

`cybernetics.sim` owns asset packaging, import, preview, render, catalog, and
launch helpers. `cybernetics.robotics` owns `RobotTaskSpec`, backend adapters,
run records, policy artifacts, replay artifacts, datasets, and evaluation
records. The two compose through plain serialized asset references:
`client.sim.import_asset(...).to_asset_ref().to_dict()` can be placed in
`RobotTaskSpec.asset_refs[]` without making robotics import the sim namespace.

`SimImportResult.to_asset_ref()` returns a `simulation-asset-ref/v1` descriptor:

```python
{
    "schema_version": "simulation-asset-ref/v1",
    "ref_kind": "environment_version",  # or "local_bundle" / "catalog_asset"
    "uri": "cybernetics://envs/env_.../versions/ver_...",
    "env_id": "env_...",
    "version_id": "ver_...",
    "root_stage_relpath": "scene.usd",
    "asset_kind": "usd_stage",
    "compatibility_status": "ready_to_render",
    "content_sha256": "...",
    "metadata": {},
}
```

RobotTask code consumes that value as an opaque descriptor:

```python
from cybernetics.robotics import RobotTaskSpec

task_payload = build_robot_task_payload(...)
task_payload["asset_refs"] = [sim_asset_ref]
task = RobotTaskSpec.from_dict(task_payload)
```

The current MVP renders USD-family assets (`.usd`, `.usda`, `.usdc`, `.usdz`)
by creating an environment version and starting a hosted session. URDF, Xacro,
SDF, and MJCF files are detected and packaged as `needs_conversion` until the
converter path lands. Public `/sim/<slug>` artifact pages are also future work;
`cybernetics sim render --public` fails explicitly instead of pretending to
publish a durable public artifact.

### Gaussian splats

Gaussian splats are first-class local assets. `.spz`/`.splat`/`.ksplat` are
detected for inspection and packaging; `.ply` is header-sniffed for the 3DGS
vertex properties so a photogrammetry mesh PLY is not misclassified. The hosted
conversion endpoint currently accepts only standard 3DGS `.ply` and emits a
validated OpenUSD `ParticleField3DGaussianSplat` USDZ (no COLMAP or training):

```bash
# upload + convert to ParticleField USDZ + wait for the artifact
cybernetics splat upload ./construction_site.ply --convert --wait

# inspect/import also understand splats (uploads a needs_conversion bundle)
cybernetics sim inspect ./construction_site.ply --format json
cybernetics sim import ./construction_site.ply --name construction-site

# then import + launch the exported USDZ like any USD asset
cybernetics sim launch ./construction_site.usdz --wait
```

`cybernetics splat upload --wait` prints a presigned `usdz_download_url` when
the conversion job completes; `cybernetics splat status <job_id>` polls an
existing job. The upload command rejects other splat containers locally, before
presigning or starting paid compute. Hosted PLY conversion is bounded to 256 MiB,
1,000,000 Gaussians, 62 scalar float properties, and a 64 KiB header; the SDK
validates the full standard 3DGS property/SH contract locally before upload.

Uploaded bundle manifests intentionally avoid host-local absolute source paths.
They keep safe provenance only: source basename, optional user-supplied
`--source-url`, root stage, asset kind, compatibility status, and per-file
hashes/sizes.

`cybernetics sim render` is preview/evidence plumbing. It answers whether an
asset can be imported, launched, and visually inspected. Robot task success,
policy evaluation, rollout records, replay evidence, datasets, and VLA/eval
records remain owned by `cybernetics.robotics`.

## RobotTask SDK contracts

`cybernetics.robotics` provides dependency-light contracts and helper adapters
for robot workflows:

- `RobotTaskSpec` for task definitions and simulator backend configuration
- `RobotEnv` / `StepResult` for backend adapter shape
- `RobotRunRecord` for rollout records
- `PolicyArtifact` for policy/checkpoint metadata
- `TrajectoryDatasetArtifact`, replay helpers, VLA eval records, world-model
  artifact metadata, and provider templates such as Unitree G1

The base robotics package is designed to import without sim, Isaac, ROS2,
MuJoCo, Worldlines, or Cosmos runtime packages installed. Heavy backend
execution belongs behind backend adapters, not in the package import path.

## PI0 DROID inference

`pi0-droid` is a frozen hosted policy. It accepts one typed raw DROID
observation and returns one action chunk with shape `[H, 8]`: seven absolute
joint-position targets followed by one gripper target. Authenticate with
`cybernetics auth login` or `CYBERNETICS_API_KEY`, then sample it through the
normal `ServiceClient`:

```python
import cybernetics
from cybernetics import types

observation = types.DroidObservation.from_numpy(
    exterior_image_0_left=exterior_rgb,
    exterior_image_1_left=second_exterior_rgb,
    wrist_image_left=wrist_rgb,
    joint_position=joint_position,
    gripper_position=gripper_position,
    instruction="pick up the cube",
)

service = cybernetics.ServiceClient(project_id="robotics-lab")
sampler = service.create_sampling_client(base_model="pi0-droid", timeout=900)
result = sampler.sample_droid(observation).result(timeout=900)
actions = result.action_chunk.to_numpy()
```

An external DSRL controller can steer the frozen policy by supplying one
finite `float32[32]` action. The SDK repeats it across PI0's ten-step initial
flow-noise horizon, snapshots the exact little-endian wire hash before
submission, and rejects the response unless the runtime acknowledges that same
`float32[10,32]` tensor:

```python
dsrl_action = types.Pi0DroidDsrlAction.from_numpy(controller_action)
result = sampler.sample_droid(observation, dsrl_action=dsrl_action).result(timeout=900)
```

The DSRL optimizer and replay buffer remain client-side; the hosted PI0 weights
stay immutable. This endpoint produces exactly one native-policy sample per
call. It does not support SDE trajectories, predicted video, LoRA/full training,
`forward_backward`, or `optim_step`. Do not confuse `pi0-droid` with the
separate trainable `pi0.5` backend. A complete NPZ-to-action-file program is in
[`examples/pi0_droid_sampling.py`](examples/pi0_droid_sampling.py).

## DreamZero examples

The repository ships an SDK-native DreamZero SFT smoke at
`examples/dreamzero_sft_smoke.py`. Its default mode is local-only and safe for
CI:

```bash
python examples/dreamzero_sft_smoke.py
python examples/dreamzero_rl_smoke.py
```

Run `cybernetics doctor` first, then add `--remote-run` only when you want the
example to create a hosted session/model, run `forward_backward`, apply
`optim_step`, and save a checkpoint. Remote examples cancel their session on
exit by default; pass `--keep-lease` only when you deliberately want to debug
inside the paid container. Use `--timeout` to bound cold-start/model-create
waits. During hosted startup, SDK queue logs surface sanitized provider progress
and worker-heartbeat waits when the control plane has that detail; once the
backend accepts work, a cold DreamZero `create_model` can also report active
`register_model` while Wan/DreamZero assets hydrate and load.

For RL readiness, use:

```bash
cybernetics doctor --require-rl
python examples/dreamzero_sft_smoke.py --remote-run --timeout 2400 --cleanup-timeout 300
python examples/dreamzero_rl_smoke.py --remote-run --timeout 2400 --cleanup-timeout 300
```

The RL smoke sends a tiny synthetic DreamZero trajectory through `flow_rwr`.
That validates the same Tinker-compatible `TrainingClient.forward_backward`
surface as SFT while exercising the DreamZero RL loss path. On a cold backend
image the first run may spend several minutes pulling and extracting the hosted
backend image before the worker heartbeat appears, then hydrate the
Wan/DreamZero asset cache and load sharded model weights. Use a timeout large
enough for that first run, and use `cybernetics doctor --require-rl` before
spending GPU time.

DreamZero sampling is a continuous-policy rollout, not token generation. Use a
normal `SamplingClient`, but send policy conditioning tensors and read
continuous artifacts from the response:

```python
sampler = service_client.create_sampling_client(base_model="dreamzero-droid")
conditioning = {
    "images": types.TensorData.from_numpy(rgb_frames),       # uint8 [B,T,H,W,3]
    "state": types.TensorData.from_numpy(proprio_state),     # float32 [B,T,D]
    "state_mask": types.TensorData.from_numpy(state_mask),   # bool [B,T,D]
    "embodiment_id": types.TensorData.from_numpy(embodiment),# int64 [B]
}
result = sampler.sample(
    types.ModelInput.empty(),
    1,
    types.SamplingParams(max_tokens=1),
    conditioning=conditioning,
).result()
actions = result.action_chunk
trajectory = result.trajectory
future_video = result.predicted_video or result.video
```

For base-model sampling the SDK omits `model_path`; do not send
`model_path: null`. Token-only clients may ignore `action_chunk`, `trajectory`,
and video fields, but VLA callers should treat those as the primary result.

## Optional dependencies

| Extra | Adds | Needed for |
|---|---|---|
| `tokenizers` | `transformers` | `get_tokenizer()` |
| `aiohttp` | `aiohttp` | the aiohttp transport |
| `torch` | `torch` | local tensor interop |
| `all` | all of the above | everything |

`numpy` is a core dependency. `transformers`, `aiohttp`, and `torch` are not —
install the relevant extra (`pip install 'cybernetic-physics[tokenizers]'`) when
you need them.
