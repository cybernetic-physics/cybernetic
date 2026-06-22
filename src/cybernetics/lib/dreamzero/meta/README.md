# DreamZero bundled metadata

`relative_stats_dreamzero.json` (the q99 relative-action statistics used by
`cybernetics.lib.dreamzero.relative_actions`) is **not** committed to this repo —
it is produced by the hosted DreamZero data pipeline and dropped in here at
release time so the wheel can ship it.

Schema: a JSON object keyed by relative-action key (e.g. `"joint_position"`),
each value carrying `q01/q99/mean/std/min/max` lists of length `len(joint_slice)`.

Until the file is present:

- `load_relative_stats(path=...)` works with an explicit path you supply, and
- `load_relative_stats()` (no arg) raises `RelativeStatsUnavailable` with an
  actionable message.

The SDK never fabricates these statistics.
