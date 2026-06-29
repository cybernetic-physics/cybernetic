# Checkpoints

A *checkpoint* is a saved set of weights from a training run — either a **training**
checkpoint (the canonical weights, including optimizer state when present) or a
**sampler** checkpoint (the weights an inference deployment loads). This page is a
task recipe for finding, inspecting, sharing, expiring, downloading, and deleting
checkpoints, both from Python (the `RestClient`) and from the CLI
(`cybernetics checkpoint ...`).

If you just want to point a sampler at a checkpoint, see [Sampling & Inference](./sampling.md).
For how checkpoints are produced, see [Training](./training.md). For the concept
of training runs vs. sessions, see the [mental model](../concepts/mental-model.md).

## The blessed path

Use the hand-written `RestClient` (obtained from a `ServiceClient` via
`create_rest_client()`, shown below) or the `cybernetics` CLI. Both are documented
below and are the supported public surface.

Do **not** call the Stainless-generated transport (`AsyncCybernetics`,
`cybernetics.resources.*`, `client.weights.list(...)`, etc.) directly — `RestClient`
wraps it, adds retries and telemetry, and is the only layer we keep stable. You will
see those internals referenced inside the SDK source; treat them as private.

You get a `RestClient` from a `ServiceClient`:

```python
from cybernetics import ServiceClient

service_client = ServiceClient()              # resolves your API key (see Authentication)
rest_client = service_client.create_rest_client()
```

Authentication is shared with the rest of the SDK: the key is resolved from
`CYBERNETICS_API_KEY`, then `CP_API_KEY`, then the deprecated `WORLDLINES_API_KEY`,
then a stored CLI login. Keys look like `cp_live_...` (CP = Cybernetic Physics). See
[Authentication](./authentication.md) for the full resolution order.

> Every `RestClient` method shown here is **synchronous-with-a-Future**: it returns
> a `concurrent.futures.Future` (or an awaitable `APIFuture`), and you call
> `.result()` to block for the value. Each sync method also has an `_async`
> counterpart (`list_checkpoints_async`, `publish_checkpoint_from_worldlines_path_async`,
> ...) that you `await` directly. The examples use the sync `.result()` form.

## `worldlines://` checkpoint URIs

Every checkpoint has a stable URI. The scheme is `worldlines://` — "Worldlines" is
the legacy product name that survives in these paths (the product/SDK is now
Cybernetics; the path scheme was not renamed). The shape is:

```text
worldlines://<training_run_id>/<kind>/<checkpoint_id>
```

where `<kind>` is `weights` for a training checkpoint or `sampler_weights` for a
sampler checkpoint. Examples:

```text
worldlines://run-123/weights/0001
worldlines://run-123/weights/final
worldlines://run-123/sampler_weights/0001
```

Parsing is done by `ParsedCheckpointCyberneticsPath` (also re-exported at the
package root). It is strict: the path must start with `worldlines://`, have at least
three segments, and use `weights` or `sampler_weights` as the second segment.

```python
from cybernetics import ParsedCheckpointCyberneticsPath

parsed = ParsedCheckpointCyberneticsPath.from_worldlines_path(
    "worldlines://run-123/weights/0001"
)
parsed.training_run_id   # "run-123"
parsed.checkpoint_type   # "training"  (weights -> training, sampler_weights -> sampler)
parsed.checkpoint_id     # "weights/0001"  (the kind + remaining segments)
```

A malformed path raises `ValueError`:

```python
ParsedCheckpointCyberneticsPath.from_worldlines_path("run-123/weights/0001")
# ValueError: Invalid worldlines path: run-123/weights/0001
```

The `*_from_worldlines_path` `RestClient` methods accept these URIs directly and
resolve the run ID / checkpoint ID for you (via a `weights_info` lookup), so you
rarely need to parse paths yourself.

## Listing checkpoints

### For one training run

`list_checkpoints(training_run_id)` returns a `CheckpointsListResponse`. This call
is **not paginated** — it returns every checkpoint for that run (training and
sampler) in one response, and `cursor` is `None`.

```python
resp = rest_client.list_checkpoints("run-123").result()
for c in resp.checkpoints:
    print(c.checkpoint_type, c.checkpoint_id, c.worldlines_path, c.public)
```

Each `Checkpoint` has these fields (from `cybernetics.types.Checkpoint`):

| Field             | Type                          | Notes |
|-------------------|-------------------------------|-------|
| `checkpoint_id`   | `str`                         | e.g. `weights/0001` |
| `checkpoint_type` | `"training"` \| `"sampler"`   | |
| `time`            | `datetime`                    | creation time |
| `worldlines_path` | `str`                         | the `worldlines://...` URI |
| `size_bytes`      | `int \| None`                 | `None` if unknown |
| `public`          | `bool`                        | defaults to `False` |
| `expires_at`      | `datetime \| None`            | `None` = never expires |

### Across all of your runs

`list_user_checkpoints(limit=100, offset=0)` lists checkpoints across **all** your
training runs, newest first, and **is** paginated — the response carries a `Cursor`
with `offset`, `limit`, and `total_count`.

```python
resp = rest_client.list_user_checkpoints(limit=50).result()
print(len(resp.checkpoints), "of", resp.cursor.total_count if resp.cursor else "?")

# Next page:
if resp.cursor and resp.cursor.offset + resp.cursor.limit < resp.cursor.total_count:
    page2 = rest_client.list_user_checkpoints(limit=50, offset=50).result()
```

> Heads-up: the cross-run query is slow server-side. The CLI fetches it in batches
> of 1000 with a progress bar; if you paginate yourself, expect each page to take a
> while.

### CLI

```bash
# All recent runs (default --limit 20)
cybernetics checkpoint list

# A specific run (returns everything for that run)
cybernetics checkpoint list --run-id run-123

# Show all of your checkpoints (no cap)
cybernetics checkpoint list --limit 0
```

`--limit` only applies when listing across runs; `--run-id` is always exhaustive.
Table output looks like:

```text
20 checkpoints (137 more not shown, use --limit to see more)
Checkpoint ID   Type      Size      Public  Created           Expires  Path
weights/0001    training  1.2 GiB   No      2026-06-20 14:03  Never    worldlines://run-123/weights/0001
sampler_weights/0001  sampler  640 MiB  Yes  2026-06-20 14:05  Never  worldlines://run-123/sampler_weights/0001
```

Add `--format json` (a global flag, placed before the subcommand:
`cybernetics --format json checkpoint list`) to get machine-readable output for an
agent to parse.

## Inspecting one checkpoint

### CLI

```bash
cybernetics checkpoint info worldlines://run-123/weights/0001
```

```text
Checkpoint: weights/0001
Property           Value
Checkpoint ID      weights/0001
Type               training
Cybernetics Path   worldlines://run-123/weights/0001
Size               1.2 GiB
Public             No
Created            2026-06-20 14:03
Expires            Never
Training Run ID    run-123
LoRA               Yes (Rank 16)
```

`info` makes two calls under the hood: it finds the checkpoint via
`list_checkpoints`, then calls `get_training_run_by_worldlines_path` to add LoRA
details. If the path is not a `worldlines://` URI the command fails immediately with
a format hint; if no checkpoint with that exact path exists you get
`Checkpoint not found: <path>`.

### Python

There is no single "get checkpoint" method. Reconstruct `info` from the building
blocks:

```python
path = "worldlines://run-123/weights/0001"

# LoRA / base-model metadata for the owning run:
run = rest_client.get_training_run_by_worldlines_path(path).result()
print(run.base_model, run.is_lora, run.lora_rank)

# The checkpoint record itself — filter the run's list by path:
resp = rest_client.list_checkpoints(run.training_run_id).result()
ckpt = next(c for c in resp.checkpoints if c.worldlines_path == path)
print(ckpt.size_bytes, ckpt.public, ckpt.expires_at)
```

You can also fetch a run directly by ID with `get_training_run("run-123")`. Both
`get_training_run` and `get_training_run_by_worldlines_path` take
`access_scope="owned"` (default) or `"accessible"` — use `"accessible"` to look up a
run you don't own but can reach (e.g. a published one). The returned `TrainingRun`
includes `training_run_id`, `base_model`, `model_owner`, `is_lora`, `lora_rank`,
`corrupted`, `last_request_time`, `last_checkpoint`, `last_sampler_checkpoint`, and
`user_metadata`.

For just the loading metadata (no full run record), use
`get_weights_info_by_worldlines_path(path)`, which returns a `WeightsInfoResponse`
with `base_model`, `is_lora`, `lora_rank`, `train_unembed`, `train_mlp`,
`train_attn`, and `has_optimizer_state`.

## Publishing and unpublishing

Publishing makes a checkpoint publicly accessible (so others can reach it with
`access_scope="accessible"`). **Only the exact owner of the training run can publish
or unpublish.**

### CLI

```bash
cybernetics checkpoint publish   worldlines://run-123/weights/0001
cybernetics checkpoint unpublish worldlines://run-123/weights/0001
```

### Python

```python
rest_client.publish_checkpoint_from_worldlines_path(
    "worldlines://run-123/weights/0001"
).result()

rest_client.unpublish_checkpoint_from_worldlines_path(
    "worldlines://run-123/weights/0001"
).result()
```

Failure modes (raised as API errors; the CLI maps them to readable messages — see
[Errors & Troubleshooting](../reference/errors.md)):

- **400** — invalid checkpoint identifier.
- **404** — checkpoint not found, or you don't own the run.
- **409** — already public (on publish) / already private (on unpublish).
- **500** — server-side error.

## Setting a TTL (expiration)

A checkpoint can be given a time-to-live, after which it expires. `ttl_seconds` is
counted **from now**; passing `None` removes any existing expiration. Owner-only.

### Python

```python
# Expire 24 hours from now:
rest_client.set_checkpoint_ttl_from_worldlines_path(
    "worldlines://run-123/weights/0001", 86400
).result()

# Remove the expiration (keep forever):
rest_client.set_checkpoint_ttl_from_worldlines_path(
    "worldlines://run-123/weights/0001", None
).result()
```

A non-positive `ttl_seconds` is rejected with **400**; **404** means not found or
not yours.

### CLI

```bash
cybernetics checkpoint set-ttl worldlines://run-123/weights/0001 --ttl 86400
cybernetics checkpoint set-ttl worldlines://run-123/weights/0001 --remove
```

You must pass exactly one of `--ttl <seconds>` or `--remove`. Passing neither, or
both, fails locally before any request:

```text
Must specify either --ttl <seconds> or --remove
Use --ttl to set an expiration or --remove to clear it
```

## Downloading a checkpoint

### CLI (download + extract)

`cybernetics checkpoint download` fetches the archive, extracts it, and deletes the
`.tar` on success. The extraction directory is named after the checkpoint:
`worldlines://` is stripped and `/` becomes `_`.

```bash
# Creates ./run-123_weights_final/ in the current directory
cybernetics checkpoint download worldlines://run-123/weights/final

# Extract under ./models/
cybernetics checkpoint download worldlines://run-123/weights/final --output ./models/

# Overwrite an existing extraction directory
cybernetics checkpoint download worldlines://run-123/weights/final --force
```

If the target directory already exists and you didn't pass `--force`:

```text
Target directory already exists: run-123_weights_final
Use --force to overwrite or choose a different output directory.
```

Building the archive can take a while; while it waits the SDK logs
`Creating checkpoint archive ... (this may take a while)` and then `... still
running`. Extraction is hardened against path-traversal and symlink/hardlink members
in the tar — a malicious or corrupt archive is rejected with "Unsafe path in tar
archive".

### Python (get the signed URL, fetch it yourself)

`RestClient` does not download bytes for you — it hands you a short-lived signed URL
and you fetch it with any HTTP client.

```python
url_resp = rest_client.get_checkpoint_archive_url_from_worldlines_path(
    "worldlines://run-123/weights/final"
).result()

print(url_resp.url)       # signed download URL
print(url_resp.expires)   # datetime when the URL expires

import urllib.request
urllib.request.urlretrieve(url_resp.url, "checkpoint.tar")
# then extract checkpoint.tar yourself
```

If you already know the run ID and checkpoint ID, the lower-level
`get_checkpoint_archive_url(training_run_id, checkpoint_id)` returns the same
`CheckpointArchiveUrlResponse` (`url`, `expires`).

## Exporting to the Hugging Face Hub

`cybernetics checkpoint push-hf` downloads the checkpoint, validates that it is a
**PEFT/LoRA adapter**, optionally writes a model card, and uploads it to a Hugging
Face repo. This is a LoRA-adapter export path: the archive must contain
`adapter_config.json`, `adapter_model.safetensors` (or `adapter_model.bin`), and a
`checkpoint_complete` sentinel. A full fine-tune archive will be rejected:

```text
Checkpoint archive does not contain a PEFT adapter.
Expected adapter_config.json and adapter_model.safetensors (or adapter_model.bin).
```

### Authentication

This command authenticates to Hugging Face through `huggingface_hub`, not through an
inline token. Install the dependency and log in first:

```bash
pip install huggingface_hub
hf auth login
```

If `huggingface_hub` is missing you get an install hint; if you are not logged in you
get `Not logged in / Run: hf auth login`. (The standalone example script in the repo,
`examples/push_checkpoint_to_hf.py`, instead reads a **`HF_TOKEN`** write token from
the environment and never accepts it on the command line — see the note at the end of
this page.)

### Usage

```bash
# Push to an explicit repo (private by default)
cybernetics checkpoint push-hf worldlines://run-123/sampler_weights/0001 \
    --repo your-org/my-lora-adapter

# Public repo, custom commit, open a PR instead of pushing to main
cybernetics checkpoint push-hf worldlines://run-123/sampler_weights/0001 \
    --repo your-org/my-lora-adapter \
    --public \
    --commit-message "Export step 0001" \
    --create-pr
```

Flags:

| Flag | Meaning |
|------|---------|
| `--repo`, `-r` | Target repo ID. If omitted, derived as `worldlines-<base_model>-<run_id>` (sanitized), with the revision defaulting to the checkpoint ID. |
| `--public` | Create/upload to a public repo. Default is **private**. |
| `--revision` | Target branch/revision to upload to. |
| `--commit-message` | Commit message for the upload. |
| `--create-pr` | Open a pull request instead of pushing to the main branch. |
| `--allow-pattern` | Only upload files matching this glob (repeatable). |
| `--ignore-pattern` | Skip files matching this glob (repeatable). |
| `--no-model-card` | Do not generate a `README.md` model card if one is missing. |

Behavior worth knowing:

- Unless you pass `--allow-pattern`, the internal `checkpoint_complete` sentinel is
  added to the ignore list so it is not uploaded.
- The generated model card embeds the source `worldlines://...` path. On re-upload,
  the command reads the existing repo `README.md`; if it references a *different*
  Cybernetics checkpoint, the push aborts with
  "Repo ID appears to contain a different Cybernetics checkpoint." This guards against
  clobbering an unrelated repo.
- The repo is created with `exist_ok=True`, so pushing to an existing repo you own is
  fine.

There is **no Python `RestClient` method** for HF export — `push-hf` is CLI-only. The
equivalent in code is: call `get_checkpoint_archive_url_from_worldlines_path`,
download and extract the tar, then use `huggingface_hub` yourself (which is exactly
what the example script does).

## Deleting checkpoints

> **Deletion is permanent and cannot be undone.** Owner-only.

### CLI

Delete by explicit path(s), or by run with optional filters:

```bash
# One or more explicit paths
cybernetics checkpoint delete worldlines://run-123/weights/0001 worldlines://run-123/weights/0002

# Every checkpoint for a run
cybernetics checkpoint delete --run-id run-123

# Filtered: only training checkpoints created before a date (UTC)
cybernetics checkpoint delete --run-id run-123 --type weights --before 2026-06-01

# A date range
cybernetics checkpoint delete --run-id run-123 --after 2026-01-01 --before 2026-02-01

# Skip the confirmation prompt (for scripts/agents)
cybernetics checkpoint delete --run-id run-123 --yes
```

Rules enforced before anything is deleted:

- You must pass either explicit paths **or** `--run-id`, not both.
- `--type`, `--before`, and `--after` require `--run-id`.
- `--type` accepts `weights` or `sampler_weights` (mapped to `training` / `sampler`).
- Dates are ISO 8601, interpreted as **UTC** (`2026-06-01` or `2026-06-01T08:00:00Z`).

Without `--yes` you get a summary and a confirmation prompt:

```text
Will delete 2 checkpoint(s):
  - worldlines://run-123/weights/0001 (1.2 GiB, created 2026-06-20 14:03)
  - worldlines://run-123/weights/0002 (1.2 GiB, created 2026-06-21 09:10)

Total size: 2.4 GiB
WARNING: This action is permanent and cannot be undone.
Are you sure you want to delete 2 checkpoint(s)? [y/N]:
```

Deletes run concurrently (up to 32 at a time). Failures are reported per-path without
aborting the rest:

```text
Deleted 1 checkpoint(s)
Failed to delete 1 checkpoint(s):
  - worldlines://run-123/weights/0002: <error>
```

### Python

```python
# By path (resolves run/checkpoint IDs for you):
rest_client.delete_checkpoint_from_worldlines_path(
    "worldlines://run-123/weights/0001"
).result()

# By explicit IDs:
rest_client.delete_checkpoint("run-123", "weights/0001").result()
```

There is no bulk/filtered delete in `RestClient` — the CLI's `--run-id`/`--type`/
`--before`/`--after` behavior is implemented client-side by listing, filtering, and
deleting each path. To replicate it, list with `list_checkpoints`, filter on
`checkpoint_type` / `time`, and loop over `delete_checkpoint_from_worldlines_path`.

## Method reference (quick map)

| Task | `RestClient` method | CLI |
|------|---------------------|-----|
| List one run's checkpoints | `list_checkpoints(run_id)` | `checkpoint list --run-id` |
| List all your checkpoints | `list_user_checkpoints(limit, offset)` | `checkpoint list` |
| Run / LoRA metadata | `get_training_run` / `get_training_run_by_worldlines_path` | `checkpoint info` |
| Loading metadata only | `get_weights_info_by_worldlines_path` | — |
| Signed download URL | `get_checkpoint_archive_url[_from_worldlines_path]` | `checkpoint download` |
| Publish | `publish_checkpoint_from_worldlines_path` | `checkpoint publish` |
| Unpublish | `unpublish_checkpoint_from_worldlines_path` | `checkpoint unpublish` |
| Set/remove TTL | `set_checkpoint_ttl_from_worldlines_path` | `checkpoint set-ttl` |
| Delete | `delete_checkpoint[_from_worldlines_path]` | `checkpoint delete` |
| Export to HF | *(none — use `huggingface_hub`)* | `checkpoint push-hf` |

Each sync method has an `_async` variant; see the full signatures in the
[Client API reference](../reference/client-api.md) and the CLI flags in the
[CLI reference](../reference/cli.md).

## Notes and limits

- **`cybernetics` vs `worldlines` in command help.** The installed CLI entry point is
  `cybernetics`. Some built-in help strings still say `worldlines checkpoint ...` —
  that is the legacy name; run the commands as `cybernetics checkpoint ...`.
- **Two different HF export paths.** The supported, archive-aware path is
  `cybernetics checkpoint push-hf` (auth via `hf auth login`, LoRA-adapter only). The
  repo also ships `examples/push_checkpoint_to_hf.py`, a standalone script that
  uploads an **already-extracted on-disk weights directory** (`model.safetensors`,
  `config.json`, `metadata.pt`, `COMPLETE` sentinel) and reads an `HF_TOKEN` write
  token from the environment. Use the CLI for adapters pulled straight from a
  `worldlines://` path; use the script when you already have the weights on disk.
- **Owner-only mutations.** publish, unpublish, set-ttl, and delete all require that
  you own the training run; otherwise expect a 404.
- **Signed URLs expire.** `CheckpointArchiveUrlResponse.expires` tells you when —
  fetch promptly.
