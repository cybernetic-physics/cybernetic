"""Renderers turn a :class:`BehaviorCiResult` into a PR comment and a report.

Renderers are presentation-only: they read the result + artifact-relative paths
and must never re-run evaluation. The comment carries a stable marker so CI can
upsert a single sticky comment.
"""

from __future__ import annotations

from typing import List

from .schemas import BehaviorCiResult

COMMENT_MARKER = "<!-- cybernetic-behavior-ci -->"
# Stable placeholder the workflow substitutes with the inline replay GIF once it has been
# published to an inline-capable URL. Anchoring to this HTML token (not a visible header)
# keeps the image injection robust against header/wording changes.
REPLAY_TOKEN = "<!-- cybernetic-behavior-ci:replay -->"
_TAGLINE = (
    "CodeRabbit reviews whether the code looks right. "
    "Cybernetic Physics reviews whether the robot still works."
)


def replay_image_block(url: str, commit: str, camera: str) -> str:
    """The SDK-owned markdown for an inline replay GIF (kept here so its format is tested)."""
    return (
        f"![Real Isaac G1 weld-approach replay — commit {commit}]({url})\n"
        f"_{camera} · settled pass/fail camera_"
    )


def render_comment(
    result: BehaviorCiResult, artifact_url: str = "", replay_gif_url: str = ""
) -> str:
    s = result.summary
    verdict = "✅ **PASS**" if result.passed else "❌ **FAIL**"
    # Replay slot: substitute the inline GIF when its URL is known, else leave the invisible
    # token for `render-comment --replay-gif-url` to fill after the GIF is hosted.
    replay_slot = (
        replay_image_block(replay_gif_url, result.commit, result.camera)
        if replay_gif_url
        else REPLAY_TOKEN
    )
    lines: List[str] = [
        COMMENT_MARKER,
        replay_slot,
        f"## Cybernetic Physics — Behavior CI {verdict}",
        "",
        f"_{_TAGLINE}_",
        "",
        "| | |",
        "|---|---|",
        f"| Behavior | `{result.behavior}` |",
        f"| Robot | {result.robot} |",
        f"| World | `{result.world}` |",
        f"| Policy | `{result.policy}` |",
        f"| Commit | `{result.commit}` |",
        f"| Trials | {s.get('passed_runs')} / {s.get('total_runs')} passed |",
        "",
        "### Checks",
    ]
    for name, ok in result.checks.items():
        lines.append(f"- {'✅' if ok else '❌'} `{name}`")

    if result.failures:
        lines += ["", "### Failures"]
        for f in result.failures:
            lines.append(f"- run {f['run']} — **{f['code']}**: {f['message']}")

    lines += ["", "### Metrics"]
    for k, v in result.metrics.items():
        lines.append(f"- {k}: `{v}`")

    h = result.honesty
    lines += [
        "",
        "### Provenance",
        f"- simulator: `{h.simulator_adapter}` · replay: `{h.replay_source}`",
        f"- policy backend: `{h.policy_backend}` (real VLA: `{str(h.policy_backend_real_vla).lower()}`)",
        f"- production eval path used: `{str(h.production_eval_path_used).lower()}`",
    ]
    if h.session_id:
        lines.append(f"- isaac session: `{h.session_id}`")
    if artifact_url:
        lines += ["", f"Artifacts & replay: [{artifact_url}]({artifact_url})"]
    return "\n".join(lines) + "\n"


def render_report_html(result: BehaviorCiResult) -> str:
    s = result.summary
    state = "pass" if result.passed else "fail"
    verdict = "PASS" if result.passed else "FAIL"

    tiles = "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in result.metrics.items()
    )
    checks = "".join(
        f'<li class="{"ok" if ok else "bad"}">{name}</li>' for name, ok in result.checks.items()
    )
    failures = (
        "".join(
            f"<li><b>{f['code']}</b> (run {f['run']}): {f['message']}</li>" for f in result.failures
        )
        or "<li>None</li>"
    )

    videos = ""
    for name in ("replay-failed", "replay-passed"):
        rel = result.artifacts.get(name.replace("-", "_"))
        if rel:
            videos += (
                f"<figure><figcaption>{name}</figcaption>"
                f'<video controls preload="metadata" src="../{rel}"></video></figure>'
            )

    replay_banner = (
        '<p class="banner">⚠ Placeholder clip — NOT a real Isaac capture. Run the hosted '
        "isaac-session workflow to attach genuine replay video from the pass/fail camera.</p>"
        if result.honesty.replay_source == "fixture-generated"
        else ""
    )

    rows = "".join(
        f"<tr><td>{t.run}</td><td class='{'ok' if t.passed else 'bad'}'>"
        f"{'pass' if t.passed else 'fail'}</td>"
        f"<td>{t.metrics.get('torch_tip_distance_to_target_cm', '')}</td>"
        f"<td>{t.metrics.get('collision_count', '')}</td>"
        f"<td>{t.metrics.get('restricted_zone_intrusions', '')}</td>"
        f"<td>{t.metrics.get('elapsed_seconds', '')}</td></tr>"
        for t in result.trials
    )

    h = result.honesty
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Behavior CI — {result.policy}</title>
<style>{_report_css()}</style></head>
<body>
<header class="{state}">
  <div class="badge">{verdict}</div>
  <h1>Cybernetic Physics — Behavior CI</h1>
  <p class="tag">{_TAGLINE}</p>
  <p class="meta">{result.behavior} · {result.robot} · <code>{result.world}</code> ·
     policy <code>{result.policy}</code> · commit <code>{result.commit}</code> ·
     {s.get("passed_runs")}/{s.get("total_runs")} trials passed</p>
</header>
<main>
  <section class="tiles">{tiles}</section>
  <section><h2>Replay evidence</h2>{replay_banner}<div class="videos">{videos or "<p>No replay attached.</p>"}</div></section>
  <section class="cols">
    <div><h2>Checks</h2><ul class="checks">{checks}</ul></div>
    <div><h2>Failures</h2><ul class="fail">{failures}</ul></div>
  </section>
  <section><h2>Per-trial</h2>
    <table><thead><tr><th>run</th><th>result</th><th>torch err (cm)</th>
    <th>collisions</th><th>zone</th><th>seconds</th></tr></thead><tbody>{rows}</tbody></table>
  </section>
  <section class="prov"><h2>Provenance (honesty)</h2>
    <ul>
      <li>simulator adapter: <code>{h.simulator_adapter}</code></li>
      <li>replay source: <code>{h.replay_source}</code></li>
      <li>policy backend: <code>{h.policy_backend}</code> (real VLA: <code>{str(h.policy_backend_real_vla).lower()}</code>)</li>
      <li>production eval path used: <code>{str(h.production_eval_path_used).lower()}</code></li>
      <li>isaac session: <code>{h.session_id or "n/a"}</code></li>
      <li>artifact contract: <code>{h.artifact_contract_version}</code></li>
    </ul>
    <p class="note">{h.notes}</p>
  </section>
</main>
</body></html>
"""


def _report_css() -> str:
    return """
:root{--ok:#1a7f37;--bad:#cf222e;--bg:#0d1117;--panel:#161b22;--fg:#e6edf3;--mut:#8b949e}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:32px 24px;border-bottom:1px solid #30363d}
header.pass{background:linear-gradient(180deg,rgba(26,127,55,.18),transparent)}
header.fail{background:linear-gradient(180deg,rgba(207,34,46,.18),transparent)}
.badge{display:inline-block;font-weight:700;letter-spacing:.08em;padding:4px 12px;border-radius:6px;color:#fff}
header.pass .badge{background:var(--ok)}header.fail .badge{background:var(--bad)}
h1{margin:.4em 0 .1em;font-size:1.5rem}.tag{color:var(--mut);font-style:italic;margin:.2em 0}
.meta{color:var(--mut);font-size:.92rem}main{padding:24px;max-width:1000px;margin:0 auto}
section{margin:0 0 28px}h2{font-size:1.05rem;border-bottom:1px solid #30363d;padding-bottom:6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--panel);border:1px solid #30363d;border-radius:8px;padding:12px}
.tile .k{color:var(--mut);font-size:.78rem}.tile .v{font-size:1.3rem;font-weight:600}
.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
video{width:100%;border-radius:8px;background:#000}figcaption{color:var(--mut);font-size:.85rem;margin-bottom:4px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:24px}
ul{list-style:none;padding:0}.checks li,.fail li{padding:4px 0}
.checks li.ok::before{content:"✅ "}.checks li.bad::before{content:"❌ "}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #30363d}
td.ok{color:var(--ok)}td.bad{color:var(--bad)}
.banner{background:#9a6700;color:#fff;padding:8px 12px;border-radius:6px;font-size:.9rem;margin:0 0 12px}
.prov .note{color:var(--mut);font-size:.88rem}code{background:#161b2255;padding:1px 5px;border-radius:4px}
"""
