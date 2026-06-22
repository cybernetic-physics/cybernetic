"""IsaacSessionAdapter against a mocked control plane + MCP gateway (no network).

Asserts the documented call sequence and response parsing: create -> poll ready
-> validate camera -> run trial -> capture+download replay -> stop session.
"""

from __future__ import annotations

import base64
import json

import pytest

httpx = pytest.importorskip("httpx")
respx = pytest.importorskip("respx")

from cybernetics.behavior_ci.backends import ScriptedPolicyBackend  # noqa: E402
from cybernetics.behavior_ci.schemas import PolicyManifest, SessionConfig  # noqa: E402
from cybernetics.behavior_ci.simulators.base import SceneSpec, looks_like_mp4  # noqa: E402
from cybernetics.behavior_ci.simulators.isaac_session import (  # noqa: E402
    IsaacSessionAdapter,
    IsaacSessionError,
)

BASE = "https://cp.example"
CAM = "/World/Cameras/BehaviorCI_PassFailCamera"
MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64


def _policy():
    return ScriptedPolicyBackend().load(
        PolicyManifest.from_dict(
            {
                "schema_version": "behavior-ci-policy/v1",
                "policy_id": "g1_weld_approach_v19",
                "display_filename": "g1_weld_approach_v19.pt",
                "behavior": "g1_weld_approach",
                "robot": "g1",
                "backend": "scripted-vla-shim",
                "controller": {"clearance_margin_cm": 14.0},
            }
        )
    )


def _mcp_handler(camera_present=True):
    # Mirrors the real MCP gateway: isaac.execute_script takes code-only and
    # returns results via stdout (no structured metrics, no emit()).
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        name = body["params"]["name"]
        args = body["params"]["arguments"]
        if name == "isaac.execute_script":
            code = args["code"]
            assert "args" not in args  # real tool has no args param
            if "CAMERA_OK" in code:  # camera-presence check script
                stdout = "CAMERA_OK\n" if camera_present else "CAMERA_MISSING\n"
            else:  # trial script imports behavior_ci_env and prints the sentinel
                assert "behavior_ci_env" in code and "sys.path.insert" in code
                result = {
                    "metrics": {
                        "torch_tip_distance_to_target_cm": 1.3,
                        "collision_count": 0,
                        "restricted_zone_intrusions": 0,
                        "max_base_tilt_degrees": 1.4,
                        "elapsed_seconds": 2.5,
                    },
                    "events": [],
                    "trajectory_id": "g1-run00",
                }
                stdout = "noise\nBEHAVIOR_CI_RESULT:" + json.dumps(result) + "\n"
            data = {"status": "success", "stdout": stdout, "stderr": ""}
        elif name == "isaac.capture_video":
            data = {"status": "success", "path": args["output_path"], "bytes": len(MP4)}
        elif name == "isaac.download_artifact":
            data = {"encoding": "base64", "data": base64.b64encode(MP4).decode()}
        else:
            data = {}
        envelope = {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "content": [{"type": "text", "text": json.dumps({"ok": True, "data": data})}]
            },
        }
        return httpx.Response(200, json=envelope)

    return handler


def _scene():
    return SceneSpec(
        world="w",
        scene_env="behavior-ci-tabletop-welding",
        camera=CAM,
        robot="g1",
        env_id="env_weld_1",
    )


def _cfg():
    return SessionConfig(
        scene_env="behavior-ci-tabletop-welding",
        camera=CAM,
        env_id="env_weld_1",
        idle_timeout_minutes=30,
        ready_timeout_seconds=30,
    )


@respx.mock
def test_full_session_lifecycle() -> None:
    create = respx.post(f"{BASE}/v1/sessions").mock(
        return_value=httpx.Response(200, json={"sessionId": "sess_test"})
    )
    respx.get(f"{BASE}/v1/sessions/sess_test").mock(
        return_value=httpx.Response(200, json={"status": "running", "isaac_extension_ready": True})
    )
    stop = respx.post(f"{BASE}/v1/sessions/sess_test/stop").mock(return_value=httpx.Response(204))
    respx.post(f"{BASE}/mcp").mock(side_effect=_mcp_handler(camera_present=True))

    with IsaacSessionAdapter(
        base_url=BASE, api_key="cp_live_x", session=_cfg(), poll_interval_seconds=0
    ) as a:
        a.prepare(_scene())
        assert a.session_id == "sess_test"
        obs = a.run_trial(_policy(), 0, {"obstacle_shift_cm": 5})
        assert obs.metrics["collision_count"] == 0
        assert obs.trajectory_id == "g1-run00"
        replays = a.capture_replays(_scene(), failed_run=None, passed_run=0)

    assert create.called
    assert len(replays) == 1
    assert replays[0].source == "isaac-sim-session-video"
    assert looks_like_mp4(replays[0].data)
    assert stop.called  # session always released on context exit


@respx.mock
def test_missing_camera_raises() -> None:
    respx.post(f"{BASE}/v1/sessions").mock(
        return_value=httpx.Response(200, json={"sessionId": "sess_test"})
    )
    respx.get(f"{BASE}/v1/sessions/sess_test").mock(
        return_value=httpx.Response(200, json={"status": "running", "isaac_extension_ready": True})
    )
    respx.post(f"{BASE}/v1/sessions/sess_test/stop").mock(return_value=httpx.Response(204))
    respx.post(f"{BASE}/mcp").mock(side_effect=_mcp_handler(camera_present=False))

    with pytest.raises(IsaacSessionError, match="camera"):
        with IsaacSessionAdapter(
            base_url=BASE, api_key="cp_live_x", session=_cfg(), poll_interval_seconds=0
        ) as a:
            a.prepare(_scene())


@respx.mock
def test_ready_poll_tolerates_transient_502() -> None:
    # Cold-boot gateway blips must not kill the run (regression: a single 502 on
    # the readiness GET previously raised and failed the whole CI job).
    respx.post(f"{BASE}/v1/sessions").mock(
        return_value=httpx.Response(200, json={"sessionId": "sess_test"})
    )
    respx.get(f"{BASE}/v1/sessions/sess_test").mock(
        side_effect=[
            httpx.Response(502),
            httpx.Response(503),
            httpx.Response(200, json={"status": "running", "isaac_extension_ready": True}),
        ]
    )
    respx.post(f"{BASE}/v1/sessions/sess_test/stop").mock(return_value=httpx.Response(204))
    respx.post(f"{BASE}/mcp").mock(side_effect=_mcp_handler(camera_present=True))

    with IsaacSessionAdapter(
        base_url=BASE,
        api_key="cp_live_x",
        session=_cfg(),
        poll_interval_seconds=0,
        transient_backoff=0,
    ) as a:
        a.prepare(_scene())
        assert a.session_id == "sess_test"
