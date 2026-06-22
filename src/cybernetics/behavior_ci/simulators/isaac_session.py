"""Hosted Isaac Sim simulator adapter.

Drives a real Cybernetic Physics Isaac session over the documented HTTP + MCP
contract — no browser, no CUA:

    POST /v1/sessions {envId,...}        -> boot a session from a saved env
    GET  /v1/sessions/:id                -> poll until running + bridge ready
    POST {mcp}/mcp  tools/call isaac.*   -> drive the scene over the bridge
      isaac.get_scene_info               -> assert the pass/fail camera exists
      isaac.execute_script               -> run the trial controller, read metrics
      isaac.capture_video                -> record from the named camera
      isaac.download_artifact            -> pull the MP4 back (base64)
    POST /v1/sessions/:id/stop           -> always release the session

Raw MCP JSON-RPC is private to this module; callers use the SimulatorAdapter
surface. Credentials are never written into artifacts.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, List, Optional

from ..backends.base import LoadedPolicy
from ..schemas import Event, TrialObservation
from .base import ReplayResult, SceneSpec, looks_like_mp4

_MEDIA_DIR = "/data/workspace/media"
_REPLAY_MAX_BYTES = 25 * 1024 * 1024
_TERMINAL_BAD = {"failed", "terminated", "stopped", "error", "snapshot_failed"}
# Transient HTTP statuses to retry/tolerate (gateway blips during cold boot).
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class IsaacSessionError(Exception):
    """A hosted session/MCP call failed (maps to runner infra exit code 3)."""


class IsaacSessionAdapter:
    adapter_id = "isaac-session"
    replay_source = "isaac-sim-session-video"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        session,  # SessionConfig
        mcp_url: Optional[str] = None,
        mcp_api_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        keep_session: bool = False,
        poll_interval_seconds: float = 3.0,
        http_client: Any = None,
        replay_duration_seconds: float = 6.0,
        replay_fps: int = 24,
        transient_retries: int = 4,
        transient_backoff: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.mcp_url = (mcp_url or base_url).rstrip("/")
        self.api_key = api_key
        self.mcp_api_key = mcp_api_key or api_key
        self.cfg = session
        self.workspace_id = workspace_id
        self.keep_session = keep_session
        self.poll_interval_seconds = poll_interval_seconds
        self.replay_duration_seconds = replay_duration_seconds
        self.replay_fps = replay_fps
        self._max_retries = transient_retries
        self._retry_backoff = transient_backoff
        self.session_id: Optional[str] = None
        self._rpc_id = 0
        self._owns_client = http_client is None
        if http_client is None:
            import httpx

            http_client = httpx.Client(timeout=60.0)
        self._client = http_client

    # -- context management: guarantee the session is released ------------- #

    def __enter__(self) -> "IsaacSessionAdapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if self.session_id and not self.keep_session:
                self._stop_session()
        finally:
            if self._owns_client:
                self._client.close()

    # -- lifecycle --------------------------------------------------------- #

    def prepare(self, scene: SceneSpec) -> None:
        env_id = scene.env_id or self.cfg.env_id
        body: Dict[str, Any] = {
            "name": f"behavior-ci-{scene.scene_env}",
            "idleTimeoutMinutes": self.cfg.idle_timeout_minutes,
        }
        if env_id:
            body["envId"] = env_id
        if self.workspace_id:
            body["workspaceId"] = self.workspace_id
        if self.cfg.gpu_spec:
            body["gpuSpec"] = self.cfg.gpu_spec

        created = self._cp("POST", "/v1/sessions", json_body=body)
        self.session_id = created.get("sessionId") or created.get("id")
        if not self.session_id:
            raise IsaacSessionError(f"session create returned no id: {created}")

        self._await_ready(self.cfg.ready_timeout_seconds)

        # isaac.get_scene_info returns a pong/assets blob with no prim list, so
        # verify the pass/fail camera by querying the stage directly.
        out = self._mcp("isaac.execute_script", {"code": _camera_check_script(scene.camera)})
        if "CAMERA_OK" not in (out.get("stdout") or ""):
            raise IsaacSessionError(
                f"required pass/fail camera '{scene.camera}' not found in the loaded "
                f"environment '{env_id or scene.scene_env}'"
            )

    def _await_ready(self, timeout_seconds: float) -> None:
        # Cold boot can take minutes and the gateway may blip (502/503) mid-poll;
        # only auth failures and terminal session states are fatal here. Everything
        # else (transient HTTP, connection errors, not-ready-yet) keeps polling
        # until the deadline.
        deadline = time.monotonic() + timeout_seconds
        last = "no response yet"
        while True:
            try:
                resp = self._send(
                    "GET",
                    f"{self.base_url}/v1/sessions/{self.session_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            except IsaacSessionError as exc:
                last = str(exc)  # connection error after retries — keep polling
                resp = None
            if resp is not None:
                if resp.status_code in (401, 403):
                    raise IsaacSessionError(f"auth failed polling session ({resp.status_code})")
                if resp.status_code < 400 and resp.content:
                    info = resp.json()
                    status = str(info.get("status", "")).lower()
                    if status in _TERMINAL_BAD:
                        raise IsaacSessionError(f"session entered terminal state '{status}'")
                    if status in {"running", "idle"} and _bridge_ready(info):
                        return
                    last = f"status={status or 'unknown'}, bridge not ready"
                else:
                    last = f"HTTP {resp.status_code}"  # transient; keep polling
            if time.monotonic() >= deadline:
                raise IsaacSessionError(
                    f"session {self.session_id} not ready after {timeout_seconds}s ({last})"
                )
            time.sleep(self.poll_interval_seconds)

    def _stop_session(self) -> None:
        try:
            self._cp("POST", f"/v1/sessions/{self.session_id}/stop")
        except Exception:
            # Best-effort cleanup; never mask the original error on teardown.
            pass

    # -- trials + replay --------------------------------------------------- #

    def run_trial(
        self, policy: LoadedPolicy, run: int, scenario: Dict[str, Any]
    ) -> TrialObservation:
        entrypoint = policy.param("session_entrypoint", "behavior_ci_run_trial")
        payload = {
            "entrypoint": entrypoint,
            "controller": policy.controller,
            "run": run,
            "scenario": scenario,
            "camera": self.cfg.camera,
        }
        # isaac.execute_script takes code only (no args) and returns results via
        # stdout, so the entrypoint args are inlined into the script and the result
        # is printed under a sentinel we parse back out.
        out = self._mcp("isaac.execute_script", {"code": _trial_script(entrypoint, payload)})
        result = _parse_result(out)
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise IsaacSessionError(f"trial {run}: controller returned no metrics: {result}")
        events = [
            Event(
                run=run,
                time_seconds=float(e.get("time_seconds", 0.0)),
                code=str(e.get("code", "EVENT")),
                message=str(e.get("message", "")),
            )
            for e in result.get("events", [])
        ]
        return TrialObservation(
            run=run,
            metrics={k: v for k, v in metrics.items()},
            events=events,
            trajectory_id=str(result.get("trajectory_id", f"{policy.policy_id}-run{run:02d}")),
        )

    def capture_replays(
        self, scene: SceneSpec, failed_run: Optional[int], passed_run: Optional[int]
    ) -> List[ReplayResult]:
        wanted: List[tuple[str, Optional[int]]] = []
        if failed_run is not None:
            wanted.append(("replay-failed", failed_run))
        if passed_run is not None:
            wanted.append(("replay-passed", passed_run))
        if not wanted:
            wanted.append(("replay-passed", None))

        replays: List[ReplayResult] = []
        for name, run in wanted:
            out_path = f"{_MEDIA_DIR}/{name}.mp4"
            self._mcp(
                "isaac.capture_video",
                {
                    "output_path": out_path,
                    "camera_prim_path": scene.camera,
                    "duration_seconds": self.replay_duration_seconds,
                    "fps": self.replay_fps,
                    "replay_run": run,
                },
            )
            dl = self._mcp(
                "isaac.download_artifact", {"path": out_path, "max_bytes": _REPLAY_MAX_BYTES}
            )
            raw = dl.get("data")
            if not isinstance(raw, str):
                raise IsaacSessionError(f"{name}: download_artifact returned no base64 'data'")
            data = base64.b64decode(raw)
            if not looks_like_mp4(data):
                raise IsaacSessionError(f"{name}: captured replay is not a valid MP4")
            replays.append(
                ReplayResult(
                    name=name, data=data, source="isaac-sim-session-video", camera=scene.camera
                )
            )
        return replays

    # -- transport --------------------------------------------------------- #

    def _send(self, method: str, url: str, **kwargs: Any):
        """HTTP with transient retry (5xx/429 + connection errors, capped backoff)."""
        import httpx

        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.request(method, url, **kwargs)
            except httpx.HTTPError as exc:
                if attempt >= self._max_retries:
                    raise IsaacSessionError(f"{method} {url}: connection error: {exc}") from exc
                time.sleep(min(self._retry_backoff * (2**attempt), 15.0))
                continue
            if resp.status_code in _TRANSIENT_STATUS and attempt < self._max_retries:
                time.sleep(min(self._retry_backoff * (2**attempt), 15.0))
                continue
            return resp
        raise IsaacSessionError(f"{method} {url}: exhausted transient retries")  # pragma: no cover

    def _cp(
        self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        resp = self._send(
            method,
            f"{self.base_url}{path}",
            json=json_body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        if resp.status_code >= 400:
            raise IsaacSessionError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    def _mcp(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._rpc_id += 1
        resp = self._send(
            "POST",
            f"{self.mcp_url}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": self._rpc_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            headers={
                "Authorization": f"Bearer {self.mcp_api_key}",
                "X-Session-Id": self.session_id or "",
                "Accept": "application/json",
            },
        )
        if resp.status_code >= 400:
            raise IsaacSessionError(f"MCP {name} -> {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        if "error" in body and body["error"]:
            raise IsaacSessionError(f"MCP {name} error: {body['error']}")
        try:
            text = body["result"]["content"][0]["text"]
            payload = json.loads(text)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise IsaacSessionError(f"MCP {name}: malformed result envelope: {exc}") from exc
        if not payload.get("ok", False):
            raise IsaacSessionError(f"MCP {name} not ok: {payload}")
        return payload.get("data", {})


def _bridge_ready(info: Dict[str, Any]) -> bool:
    if info.get("isaac_extension_ready") is True:
        return True
    bridge = info.get("bridge_status") or info.get("bridgeStatus") or {}
    return bool(isinstance(bridge, dict) and bridge.get("isaac_extension_ready"))


_RESULT_SENTINEL = "BEHAVIOR_CI_RESULT:"


def _parse_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the entrypoint's JSON result out of the execute_script stdout."""
    stdout = (data.get("stdout") or "") if isinstance(data, dict) else ""
    for line in stdout.splitlines():
        if line.startswith(_RESULT_SENTINEL):
            return json.loads(line[len(_RESULT_SENTINEL) :])
    stderr = (data.get("stderr") or "") if isinstance(data, dict) else ""
    raise IsaacSessionError(
        f"no {_RESULT_SENTINEL} line in session output; "
        f"stdout={stdout[-400:]!r} stderr={stderr[-400:]!r}"
    )


def _camera_check_script(camera: str) -> str:
    """Print CAMERA_OK iff the pass/fail camera prim exists on the stage."""
    return (
        "import omni.usd\n"
        f"_p = omni.usd.get_context().get_stage().GetPrimAtPath({camera!r})\n"
        "print('CAMERA_OK' if _p.IsValid() else 'CAMERA_MISSING')\n"
    )


def _trial_script(entrypoint: str, args: Dict[str, Any]) -> str:
    """Session-side invocation. The published behavior-ci environment provides
    ``behavior_ci_env.{entrypoint}(args) -> {"metrics": {...}, "events": [...]}``.

    isaac.execute_script has no ``args`` parameter and no ``emit()``, and
    ``/data/workspace`` is not on ``sys.path`` by default — so we inline the args
    (base64 to avoid quoting), add the workspace to the path, and print the result
    under a sentinel the adapter parses from stdout.
    """
    blob = base64.b64encode(json.dumps(args).encode()).decode()
    return (
        "import sys, json, base64\n"
        "sys.path.insert(0, '/data/workspace')\n"
        "import behavior_ci_env\n"
        f"_args = json.loads(base64.b64decode('{blob}').decode())\n"
        f"_res = behavior_ci_env.{entrypoint}(_args)\n"
        f"print({_RESULT_SENTINEL!r} + json.dumps(_res))\n"
    )
