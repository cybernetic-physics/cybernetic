# Behavior CI Config

This page is the exhaustive schema reference for the three repo-authored input
files that Behavior CI reads, plus the output bundle it writes. Every field,
default, and validation rule below is taken directly from
`src/cybernetics/behavior_ci/schemas.py` (the hand-written `from_dict` parsers),
`src/cybernetics/behavior_ci/runner.py`, and
`src/cybernetics/behavior_ci/artifacts.py`.

For the task-level walkthrough ("how do I wire this into my repo and a PR
comment") see the [Behavior CI guide](../guides/behavior-ci.md). For the
client/SDK surface see the [Client API](./client-api.md). For environment
variables and key formats see [Authentication](../guides/authentication.md).

The contracts are deliberately plain dataclasses, not pydantic models, so the
emitted bundle round-trips identically across the `pydantic>=1.9,<3` range the
SDK supports and can be validated from non-Python CI. Fields may be *added*
compatibly; existing keys are never renamed.

## Schema versions at a glance

Every file carries a `schema_version` string. The parser rejects the file with a
`ConfigError` if it does not match exactly.

| File | `schema_version` value | Parser | Constant |
|---|---|---|---|
| `cybernetic-behavior-ci.yaml` | `cybernetic-behavior-ci-config/v1` | `BehaviorCiConfig.from_dict` | `CONFIG_SCHEMA_VERSION` |
| eval YAML | `behavior-ci-eval/v1` | `EvalSpec.from_dict` | `EVAL_SCHEMA_VERSION` |
| policy `.pt` manifest | `behavior-ci-policy/v1` | `PolicyManifest.from_dict` | `POLICY_SCHEMA_VERSION` |
| task spec (optional, separate) | `behavior-ci-task/v1` | `TaskSpec.from_dict` | `TASK_SCHEMA_VERSION` |
| `result.json` (output) | `behavior-ci/v1` | `BehaviorCiResult` | `RESULT_SCHEMA_VERSION` |
| `metrics.json` (output) | `behavior-ci-metrics/v1` | `_metrics_doc` | `METRICS_SCHEMA_VERSION` |
| `manifest.normalized.json` (output) | `behavior-ci-normalized/v1` | `_normalized_manifest` | (inline literal) |

> Naming note: the install package is `pip install cybernetic-physics`, but the
> import package is `cybernetics`. The product/SDK brand is **Cybernetics**;
> **CP** ("Cybernetic Physics") survives in env names like `CP_API_KEY` and the
> `cp_live_` key prefix; **Worldlines** is the legacy name that survives in
> `worldlines://` paths and the deprecated `WORLDLINES_API_KEY`. See
> [Authentication](../guides/authentication.md).

---

## 1. `cybernetic-behavior-ci.yaml` — the central repo manifest

Schema version: **`cybernetic-behavior-ci-config/v1`**. Parsed by
`BehaviorCiConfig.from_dict`. This is the file you pass to
`BehaviorCiRunner.from_config(path)`; paths inside it (eval paths, replay dir,
module path) are resolved relative to the directory that contains this file.

### Top-level keys

| Key | Required | Type | Default | Notes |
|---|---|---|---|---|
| `schema_version` | **yes** | string | — | Must equal `cybernetic-behavior-ci-config/v1`. |
| `project` | **yes** | string | — | Free-text project name; copied into `manifest.normalized.json`. |
| `robot` | **yes** | string | — | Robot description; becomes `scene.robot` and `result.robot` source. |
| `simulator` | **yes** | mapping | — | See [simulator](#simulator-block) below. Missing → `ConfigError`. |
| `evals` | **yes** | mapping | — | Named eval map. See [evals](#evals-block) below. Missing → `ConfigError`. |
| `artifacts` | no | mapping | `{}` | See [artifacts](#artifacts-block) below. |
| `policy_backends` | no | mapping → mapping | `{}` | Free-form per-backend config, passed through verbatim. Not otherwise validated by `from_dict`. |

There is no top-level `out` key; output location lives under `artifacts.out`.

### `simulator` block

`simulator` is required. Its keys:

| Key | Required | Type | Default | Notes |
|---|---|---|---|---|
| `adapter` | no | string | `"fixture"` | Must be one of `SIMULATOR_ADAPTERS` = `("fixture", "isaac-session")`. Any other value → `ConfigError`. |
| `base_url_env` | no | string | `"CYBERNETICS_BASE_URL"` | Name of the env var consulted for the base URL fallback (hosted adapter only). |
| `base_url` | no | string | `None` | Non-secret control-plane URL, checked into the repo. Hosted adapter only. |
| `mcp_url` | no | string | `None` | Non-secret MCP gateway URL. Falls back to `CYBERNETICS_MCP_URL`, then `base_url`. |
| `workspace_id` | no | string | `None` | Non-secret workspace id. Falls back to `CYBERNETICS_WORKSPACE_ID`. |
| `session` | no | mapping | `{}` | Hosted-session settings; see [session](#simulatorsession-block). |

Only API keys are secrets. URLs, workspace ids, and env ids are public config and
belong in this file, not in GitHub secrets.

For the `fixture` adapter, `base_url`/`mcp_url`/`workspace_id`/`session.*` are
parsed but largely unused — fixture mode runs deterministically with no network.

### `simulator.session` block

Maps to the frozen `SessionConfig` dataclass. Honored by the `isaac-session`
adapter; parsed (with defaults) regardless of adapter.

| Key | Required | Type | Default | Notes |
|---|---|---|---|---|
| `scene_env` | no | string | `""` | Saved scene/environment to load. Also written into result + provenance. |
| `camera` | no | string | `""` | Camera prim path used for pass/fail replay capture. |
| `env_id` | no | string | `None` | Pre-built environment id. Falls back to `BEHAVIOR_CI_ENV_ID`. |
| `gpu_spec` | no | string | `None` | Requested GPU class for the hosted session. |
| `runtime_provider` | no | string | `None` | Dedicated-instance request (e.g. `"vast"`). Only honored for callers allowed to select runtime (admins / service accounts); otherwise the platform warm pool is used. |
| `idle_timeout_minutes` | no | int | `120` | Coerced with `int()`. |
| `ready_timeout_seconds` | no | int | `900` | Coerced with `int()`. |
| `spawn_robot` | no | string | `None` | Author-at-runtime: robot to spawn on a fresh blank session. |
| `spawn_position` | no | list[float] | `None` | Spawn position for `spawn_robot`. |
| `module_path` | no | string | `None` | Repo-relative path to a Python module the adapter uploads to build the scene. Resolved against the config dir; missing file → `ConfigError`. |
| `setup_entrypoint` | no | string | `None` | Name of the setup function called inside the uploaded module. |

### `artifacts` block

| Key | Required | Type | Default | Notes |
|---|---|---|---|---|
| `out` | no | string | `"artifacts/behavior-ci"` | Bundle output directory. Used only when `run_policy` is called without an explicit `out_dir`. |
| `require_replay_provenance` | no | string or null | `None` | If set, must be one of `REPLAY_SOURCES` (below) or → `ConfigError`. Enforced after the run; mismatch → `ContractError`. |
| `replay_source_dir` | no | string | `None` | Directory of checked-in replay clips for the fixture adapter, resolved against the config dir. |

`REPLAY_SOURCES` = `("isaac-sim-session-video", "checked-in-demo-evidence",
"fixture-generated", "none")`.

Provenance enforcement has real teeth and interacts with the adapter:

- `require_replay_provenance: isaac-sim-session-video` with `adapter: fixture`
  fails fast at adapter-build time (`ConfigError`) — fixture mode cannot produce
  Isaac session video.
- `require_replay_provenance: checked-in-demo-evidence` disables the fixture
  adapter's placeholder replays (`allow_placeholder_replays=False`), forcing real
  clips from `replay_source_dir`.
- After the run, if the actual `replay_source` does not equal the required value,
  the runner raises `ContractError`.

### `evals` block

`evals` is a required mapping from an eval *name* to its YAML path. Two accepted
shapes per entry:

```yaml
evals:
  # shape A: name -> path string
  obstacle_shift: evals/g1_weld_obstacle_shift.yaml
  # shape B: name -> mapping with a required 'path' key
  reach_test:
    path: evals/g1_weld_reach.yaml
```

Shape B with no `path` key → `ConfigError: evals.<name>: missing required field
'path'`. The name is what you pass as `eval_ref` to `run_policy`; a direct path is
also accepted there and tried if the name is not in the map.

### Minimal valid config (fixture mode)

This is the exact config used in `tests/test_behavior_ci_runner.py`:

```yaml
schema_version: cybernetic-behavior-ci-config/v1
project: unitree-g1-vla-policies
robot: Unitree G1-compatible humanoid proxy
artifacts:
  out: artifacts/behavior-ci
simulator:
  adapter: fixture
  session:
    scene_env: behavior-ci-tabletop-welding
    camera: /World/Cameras/BehaviorCI_PassFailCamera
evals:
  obstacle_shift: evals/g1_weld_obstacle_shift.yaml
```

### Hosted-session config (isaac-session)

```yaml
schema_version: cybernetic-behavior-ci-config/v1
project: unitree-g1-vla-policies
robot: Unitree G1-compatible humanoid proxy
artifacts:
  out: artifacts/behavior-ci
  require_replay_provenance: isaac-sim-session-video
simulator:
  adapter: isaac-session
  base_url: https://api.cyberneticphysics.com
  # base_url_env, mcp_url, workspace_id optional; secrets stay in env / login store
  session:
    scene_env: behavior-ci-tabletop-welding
    camera: /World/Cameras/BehaviorCI_PassFailCamera
    gpu_spec: l40s
    idle_timeout_minutes: 120
    ready_timeout_seconds: 900
evals:
  obstacle_shift: evals/g1_weld_obstacle_shift.yaml
```

The hosted adapter needs an API key, resolved via `resolve_api_key()`
(`CYBERNETICS_API_KEY` or `CP_API_KEY`, or `cybernetics auth login`). Missing key
→ `ConfigError`. The base URL is resolved in order: `simulator.base_url` →
`resolve_base_url()` → `os.environ[base_url_env]` → `https://api.cyberneticphysics.com`.

---

## 2. The eval YAML — `behavior-ci-eval/v1`

Schema version: **`behavior-ci-eval/v1`**. Parsed by `EvalSpec.from_dict`. This is
the pinned behavior eval: the world, how many trials, the per-trial pass/fail
checks, and per-run scenario variation.

### Top-level keys

| Key | Required | Type | Default | Notes |
|---|---|---|---|---|
| `schema_version` | **yes** | string | — | Must equal `behavior-ci-eval/v1`. |
| `world` | **yes** | string | — | Logical world id for the eval. |
| `behavior` | **yes** | string | — | Behavior under test. Must match the policy manifest's `behavior` or the runner raises `ConfigError`. |
| `runs` | **yes** | int | — | Number of trials. Coerced with `int()`; must satisfy `1 <= runs <= 64`, else `ConfigError`. |
| `checks` | **yes** | mapping → mapping | — | Named checks; see [checks](#checks). Missing → `ConfigError`. |
| `scenarios` | no | list[mapping] | `[]` | Per-run variation, free-form. See [scenarios](#scenarios). |

### `checks`

`checks` maps a check *name* to a `Check` (frozen dataclass). Each check is one
scalar pass/fail rule applied to a per-trial metric.

| Key | Required | Type | Default | Notes |
|---|---|---|---|---|
| `metric` | **yes** | string | — | The per-trial metric key this check reads. The metric must be present in the trial's observation, else evaluation raises `KeyError`. |
| `operator` | **yes** | string | — | Must be a key of `OPS` (below), else `ConfigError`. |
| `value` | **yes** | any | — | The right-hand comparison value (number, etc.). |
| `required` | no | bool | `True` | Coerced with `bool()`. Only `required` checks affect the trial verdict; non-required checks are recorded but do not fail a trial. |

`OPS` supported operators: `==`, `!=`, `<`, `<=`, `>`, `>=`. A check passes when
`OPS[operator](actual_metric, value)` is truthy.

### `scenarios`

`scenarios` is a list of opaque mappings, one intended per run, indexed by run
number. The runner passes `spec.scenarios[run]` to the adapter for run `run`; if
there are fewer scenarios than `runs`, later runs get `{}`. The fixture model
reads keys like `required_clearance_cm` from each scenario, but the schema does
not constrain scenario contents — they are whatever your adapter understands.

### Worked eval (the golden contract)

From `tests/test_behavior_ci_evaluator.py`. This exact eval is red for the v18
policy and green for v19 — the difference is purely the policy's
`clearance_margin_cm` versus each scenario's `required_clearance_cm`.

```yaml
schema_version: behavior-ci-eval/v1
world: tabletop_welding_obstacle_shift_v1
behavior: g1_weld_approach
runs: 8
checks:
  target_reach:
    metric: torch_tip_distance_to_target_cm
    operator: "<="
    value: 2.0
  collision_free:
    metric: collision_count
    operator: "=="
    value: 0
  safety_zone_clear:
    metric: restricted_zone_intrusions
    operator: "=="
    value: 0
  base_stable:
    metric: max_base_tilt_degrees
    operator: "<="
    value: 5.0
  timeout_free:
    metric: elapsed_seconds
    operator: "<"
    value: 30
scenarios:
  - {obstacle_shift_cm: 3, required_clearance_cm: 3}
  - {obstacle_shift_cm: 5, required_clearance_cm: 5}
  - {obstacle_shift_cm: 4, required_clearance_cm: 4}
  - {obstacle_shift_cm: 11, required_clearance_cm: 11, stresses: safety_zone}
  - {obstacle_shift_cm: 6, required_clearance_cm: 6}
  - {obstacle_shift_cm: 13, required_clearance_cm: 13, stresses: collision}
  - {obstacle_shift_cm: 5, required_clearance_cm: 5}
  - {obstacle_shift_cm: 10, required_clearance_cm: 10, stresses: timeout}
```

When a required check fails on a trial but the rollout produced no explanatory
event, the evaluator synthesizes one so the PR comment always names a failure
code. The check-name → code map (from `evaluator.py`):

| Check name | Synthesized failure code |
|---|---|
| `safety_zone_clear` | `SAFETY_ZONE_INTRUSION` |
| `collision_free` | `OBSTACLE_COLLISION` |
| `timeout_free` | `TARGET_TIMEOUT` |
| `target_reach` | `TARGET_MISS` |
| `base_stable` | `BASE_INSTABILITY` |
| (any other) | `CHECK_FAILURE` |

---

## 3. The policy `.pt` manifest — `behavior-ci-policy/v1`

Schema version: **`behavior-ci-policy/v1`**. Parsed by `PolicyManifest.from_dict`.

> Honest limit: in this build a policy `.pt` file is a JSON manifest, not a
> learned-weights checkpoint. Behavior is driven by the readable `controller`
> parameters, not a hidden lookup by `policy_id`. Only the `real-vla` backend
> claims a real learned policy; every other backend must report
> `policy_backend_real_vla = false` in provenance.

| Key | Required | Type | Default | Notes |
|---|---|---|---|---|
| `schema_version` | **yes** | string | — | Must equal `behavior-ci-policy/v1`. |
| `policy_id` | **yes** | string | — | Stable id; copied to `result.policy_id`. |
| `display_filename` | **yes** | string | — | Human filename (e.g. `g1_weld_approach_v19.pt`); copied to `result.policy`. |
| `behavior` | **yes** | string | — | Must match the eval's `behavior` or the runner raises `ConfigError`. |
| `robot` | **yes** | string | — | Copied to `result.robot`. |
| `backend` | **yes** | string | — | Must be one of `POLICY_BACKENDS` = `("scripted-vla-shim", "gr00t-adapter-stub", "real-vla")`. Any other → `ConfigError`. |
| `controller` | **yes** | mapping | — | Readable behavior parameters (e.g. `clearance_margin_cm`). Stored verbatim, passed through to `manifest.normalized.json`. Missing → `ConfigError`. |
| `expected_demo_result` | no | string or null | `None` | Documentation hint only. |
| `notes` | no | string or null | `None` | Falls back to the `training_note` key if `notes` is absent. |

`PolicyManifest.real_vla` is a derived property: `True` iff `backend ==
"real-vla"`.

### Example policy manifests

From the runner test — two policies differing only in `clearance_margin_cm`:

```json
{
  "schema_version": "behavior-ci-policy/v1",
  "policy_id": "g1_weld_approach_v19",
  "display_filename": "g1_weld_approach_v19.pt",
  "behavior": "g1_weld_approach",
  "robot": "Unitree G1-compatible humanoid proxy",
  "backend": "scripted-vla-shim",
  "controller": {"type": "scripted_trajectory", "clearance_margin_cm": 14.0}
}
```

The v18 manifest is identical except `policy_id`/`display_filename` and
`clearance_margin_cm: 6.0`, which makes it fail runs 3, 5, and 7.

---

## Appendix: the optional task spec — `behavior-ci-task/v1`

`TaskSpec.from_dict` exists in `schemas.py` but is not consumed by
`BehaviorCiRunner.run_policy` in this build. Documented here for completeness; do
not rely on the runner reading it.

| Key | Required | Type | Default |
|---|---|---|---|
| `schema_version` | **yes** | string (`behavior-ci-task/v1`) | — |
| `task_id` | **yes** | string | — |
| `world` | **yes** | string | — |
| `scene_env` | **yes** | string | — |
| `robot` | **yes** | string | — |
| `behavior` | **yes** | string | — |
| `camera` | **yes** | string | — |
| `workspace` | no | mapping | `{}` |

---

## 4. The output bundle — `behavior-ci/v1`

`artifacts.write_bundle` writes the bundle; `artifacts.validate_bundle` checks it.
The runner calls `validate_bundle` after writing and raises `ContractError` if any
problem is reported.

### Bundle members

`BASE_REQUIRED` (each must exist and be non-empty):

| Path | Schema / type | Contents |
|---|---|---|
| `result.json` | `behavior-ci/v1` | Top-level verdict + provenance (`BehaviorCiResult.to_dict`). |
| `metrics.json` | `behavior-ci-metrics/v1` | Aggregate metrics, per-check pass counts, and full per-run results. |
| `comment.md` | markdown | Sticky PR comment. Contains `PASS` (green) or `FAIL` (red). |
| `report/index.html` | HTML | Self-contained static report. Fixture placeholder clips are loudly disclosed ("Placeholder clip"). |
| `manifest.normalized.json` | `behavior-ci-normalized/v1` | The effective, resolved run inputs (policy + eval + simulator + commit). |
| `provenance.json` | (honesty block) | `HonestyProvenance.to_dict`, standalone for auditing. |

Plus replay clips under `replays/`:

| Path | Written when |
|---|---|
| `replays/replay-passed.mp4` | Whenever at least one trial passed. |
| `replays/replay-failed.mp4` | Only when the run failed. An honest bundle never ships a "failed" clip for a green run. |

`write_bundle` also back-fills `result.artifacts` with relative paths. Each replay
`name` is keyed with hyphens turned to underscores (`replay-passed` →
`replay_passed`). A convenience `replay_video` key points to `replay_failed` if
present, else `replay_passed`. The non-replay links are
`result_json`/`metrics_json`/`report`.

### `result.json` keys

From `BehaviorCiResult.to_dict`:

| Key | Type | Notes |
|---|---|---|
| `schema_version` | string | `behavior-ci/v1`. |
| `status` | string | `"passed"` or `"failed"`. |
| `behavior` | string | From the eval. |
| `robot` | string | From the policy manifest. |
| `world` | string | From the eval. |
| `scene_env` | string | From `simulator.session.scene_env`. |
| `camera` | string | From `simulator.session.camera`. |
| `policy` | string | Policy `display_filename`. |
| `policy_id` | string | Policy `policy_id`. |
| `policy_backend` | string | Policy `backend`. |
| `commit` | string | `git rev-parse --short HEAD` in the config dir, or `"unknown"`. |
| `summary` | mapping | `passed_runs`, `total_runs`, `failed_runs`, `world`, `behavior`. |
| `checks` | mapping[str, bool] | Per-check aggregate: green iff that check passed in *every* trial. |
| `metrics` | mapping | Aggregate metrics (below) plus `task_success` (`"<passed> / <total>"`). |
| `failures` | list[mapping] | One entry per trial event: `{run, code, message}`. |
| `artifacts` | mapping[str, str] | Relative paths back-filled by `write_bundle`. |
| `honesty` | mapping | Provenance block (below). |

Note: the in-memory `BehaviorCiResult` also carries `trials` (full per-trial
results), but `to_dict` does *not* serialize them into `result.json` — the per-run
detail lives in `metrics.json` instead.

Aggregate `metrics` keys (`evaluator._aggregate_metrics`):
`mean_torch_tip_error_cm`, `collision_events`, `safety_zone_violations`,
`max_base_tilt_degrees`, `mean_trial_seconds`, and `task_success`.

### `provenance.json` / `honesty` keys

From `HonestyProvenance.to_dict`:

| Key | Type | Notes |
|---|---|---|
| `simulator_adapter` | string | Adapter id, e.g. `fixture` or `isaac-session`. |
| `replay_source` | string | One of `REPLAY_SOURCES`. |
| `policy_backend` | string | The policy's backend. |
| `policy_backend_real_vla` | bool | `True` only for `real-vla`. Fixture/scripted runs are `False`. |
| `production_eval_path_used` | bool | Currently always `False` from the runner. |
| `scene_env` | string | — |
| `camera` | string | — |
| `artifact_contract_version` | string | Defaults to `behavior-ci/v1`. |
| `session_id` | string or null | Hosted session id, when applicable. |
| `notes` | string | Human note; fixture mode discloses that outcomes are computed from controller parameters, not a learned policy or live Isaac. |

### `metrics.json` keys

From `artifacts._metrics_doc`:

| Key | Type | Notes |
|---|---|---|
| `schema_version` | string | `behavior-ci-metrics/v1`. |
| `aggregate` | mapping | Same as `result.metrics`. |
| `check_pass_counts` | mapping[str, int] | Per-check count of trials that passed it. |
| `runs` | list[mapping] | Full `TrialResult.to_dict` per run: `run`, `passed`, `checks`, `metrics`, `events`, `trajectory_id`. |

---

## 5. Pass/fail semantics

The verdict is computed in `evaluator.aggregate` and enforced in
`artifacts.validate_bundle`:

- **Per trial**: a trial `passed` iff *every required* check passed. Non-required
  checks (`required: false`) are recorded but never fail a trial.
- **Per check (aggregate)**: `result.checks[name]` is `True` iff that check passed
  in *every* trial (and `False` if there were no trials).
- **Overall status**: `"passed"` iff `passed_runs == total_runs` and
  `total_runs > 0`; otherwise `"failed"`. One failed trial fails the whole run.
- **`failures`**: flattened list of every trial's events (synthesized failure
  events included), as `{run, code, message}`.

### Bundle validation rules

`validate_bundle` returns a list of contract violations (empty list = valid). It
flags:

- Any `BASE_REQUIRED` member missing or zero-byte.
- No `*.mp4` in `replays/`.
- Any replay that does not pass the `looks_like_mp4` byte check.
- If a `result` is passed in and `status == "failed"` but
  `replays/replay-failed.mp4` is absent.
- If `summary.passed_runs > 0` but `replays/replay-passed.mp4` is absent.

The runner raises `ContractError` if `validate_bundle` returns any problem, so an
invalid bundle never silently ships.

### Expected outcomes from the golden tests

`tests/test_behavior_ci_runner.py` pins these exact outcomes for the obstacle
shift eval (`runs: 8`):

| Policy | `clearance_margin_cm` | `status` | `summary.passed_runs` | Failing runs / codes | `replay-failed.mp4` |
|---|---|---|---|---|---|
| `g1_weld_approach_v18` | 6.0 | `failed` | 5 | runs 3, 5, 7 → `SAFETY_ZONE_INTRUSION`, `OBSTACLE_COLLISION`, `TARGET_TIMEOUT` | present |
| `g1_weld_approach_v19` | 14.0 | `passed` | 8 | none | absent (green run ships no failed clip) |

Both runs are fixture mode: `honesty.simulator_adapter == "fixture"` and
`honesty.policy_backend_real_vla is False`, and `validate_bundle(out, result)`
returns `[]`.

---

## Failure modes (what `ConfigError` / `ContractError` look like)

These are raised by `from_dict` parsing and the runner. All derive from
`BehaviorCiError`.

| Condition | Exception | Message shape |
|---|---|---|
| Wrong/missing `schema_version` | `ConfigError` | `<where>: schema_version must be '<expected>', got <got>` |
| Missing required field | `ConfigError` | `<where>: missing required field '<key>'` |
| Unknown check operator | `ConfigError` | `check '<name>': unknown operator <op>; allowed [...]` |
| `runs` out of `[1, 64]` | `ConfigError` | `eval spec: runs must be in [1, 64], got <n>` |
| Bad `simulator.adapter` | `ConfigError` | `config: simulator.adapter must be one of [...]` |
| Bad `artifacts.require_replay_provenance` | `ConfigError` | `config: artifacts.require_replay_provenance must be one of [...]` |
| Bad policy `backend` | `ConfigError` | `policy manifest: backend must be one of [...]` |
| `evals.<name>` mapping without `path` | `ConfigError` | `evals.<name>: missing required field 'path'` |
| `fixture` adapter + `require_replay_provenance: isaac-sim-session-video` | `ConfigError` | adapter cannot produce Isaac session video |
| Policy `behavior` != eval `behavior` | `ConfigError` | `policy behavior '<a>' != eval behavior '<b>'` |
| `isaac-session` with no API key | `ConfigError` | needs `CYBERNETICS_API_KEY` / `CP_API_KEY` / `cybernetics auth login` |
| Replay provenance mismatch after run | `ContractError` | `replay provenance '<got>' does not meet the required '<req>'` |
| Bundle fails validation | `ContractError` | `artifact bundle invalid:` + bullet list |
| YAML/JSON not a mapping/object, or file missing | `ConfigError` | `<path>: expected a YAML mapping` / `file not found: <path>` |

See [Errors & Troubleshooting](./errors.md) for the broader SDK error surface and
[CLI](./cli.md) for the `cybernetics behavior-ci` commands that drive this runner.
