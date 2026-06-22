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

        scene_info = self._mcp("isaac.get_scene_info", {})
        if not _camera_present(scene_info, scene.camera):
            raise IsaacSessionError(
                f"required pass/fail camera '{scene.camera}' not found in the loaded "
                f"environment '{env_id or scene.scene_env}'"
            )

    def _await_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            info = self._cp("GET", f"/v1/sessions/{self.session_id}")
            status = str(info.get("status", "")).lower()
            if status in _TERMINAL_BAD:
                raise IsaacSessionError(f"session entered terminal state '{status}'")
            if status in {"running", "idle"} and _bridge_ready(info):
                return
            if time.monotonic() >= deadline:
                raise IsaacSessionError(
                    f"session {self.session_id} not ready after {timeout_seconds}s "
                    f"(last status '{status}')"
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
        data = self._mcp(
            "isaac.execute_script",
            {"code": _trial_script(entrypoint), "args": payload},
        )
        metrics = data.get("metrics")
        if not isinstance(metrics, dict):
            raise IsaacSessionError(f"trial {run}: controller returned no metrics: {data}")
        events = [
            Event(
                run=run,
                time_seconds=float(e.get("time_seconds", 0.0)),
                code=str(e.get("code", "EVENT")),
                message=str(e.get("message", "")),
            )
            for e in data.get("events", [])
        ]
        return TrialObservation(
            run=run,
            metrics={k: v for k, v in metrics.items()},
            events=events,
            trajectory_id=str(data.get("trajectory_id", f"{policy.policy_id}-run{run:02d}")),
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

    def _cp(
        self, method: str, path: str, json_body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        resp = self._client.request(
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
        resp = self._client.post(
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


def _camera_present(scene_info: Dict[str, Any], camera: str) -> bool:
    # The scene-info payload shape varies; accept any place the prim path shows up.
    prims = scene_info.get("cameras") or scene_info.get("prims") or scene_info.get("camera_prims")
    if isinstance(prims, list):
        return any(camera == (p if isinstance(p, str) else p.get("path")) for p in prims)
    return camera in json.dumps(scene_info)


def _trial_script(entrypoint: str) -> str:
    """Session-side invocation. The published behavior-ci environment defines
    ``entrypoint(args)`` and returns ``{"metrics": {...}, "events": [...]}``."""

    return (
        "import json, behavior_ci_env\n"
        f"result = behavior_ci_env.{entrypoint}(args)\n"
        "emit(json.dumps(result))\n"
    )
