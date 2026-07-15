from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from cybernetics.sim import (
    SessionMCPClient,
    SimulationClient,
    SimulationMCPError,
)

BASE = "https://api.test"
ROOT_KEY = "cp_live_workspace_test"
SCOPED_KEY = "cp_live_scoped_secret"
SESSION_ID = "sess_hosted"
KEY_ID = "key_scoped"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("CYBERNETICS_API_KEY", raising=False)
    monkeypatch.delenv("CYBERNETICS_BASE_URL", raising=False)
    monkeypatch.delenv("CP_API_KEY", raising=False)
    monkeypatch.delenv("CP_API_BASE", raising=False)


def _grant() -> dict[str, Any]:
    return {
        "id": KEY_ID,
        "key": SCOPED_KEY,
        "sessionId": SESSION_ID,
        "keyKind": "session",
        "ttlSeconds": 900,
    }


def _tool_envelope(request_id: int, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"ok": True, "data": data}),
                }
            ]
        },
    }


def _mock_grant_and_revoke() -> tuple[respx.Route, respx.Route]:
    mint = respx.post(f"{BASE}/v1/api-keys/session-scoped").mock(
        return_value=httpx.Response(201, json=_grant())
    )
    revoke = respx.delete(f"{BASE}/v1/api-keys/{KEY_ID}").mock(return_value=httpx.Response(204))
    return mint, revoke


@respx.mock
def test_mcp_session_mints_private_scoped_key() -> None:
    mint, revoke = _mock_grant_and_revoke()

    with SimulationClient(api_key=ROOT_KEY, base_url=BASE) as client:
        mcp = client.mcp_session(SESSION_ID, ttl_seconds=900, name="sdk-test")
        assert isinstance(mcp, SessionMCPClient)
        assert mcp.session_id == SESSION_ID
        assert SCOPED_KEY not in repr(mcp)
        assert not hasattr(mcp, "api_key")
        mcp.close()

    request = mint.calls[0].request
    assert request.headers["Authorization"] == f"Bearer {ROOT_KEY}"
    assert json.loads(request.content) == {
        "sessionId": SESSION_ID,
        "ttlSeconds": 900,
        "name": "sdk-test",
    }
    assert revoke.call_count == 1
    assert revoke.calls[0].request.headers["Authorization"] == f"Bearer {ROOT_KEY}"


@respx.mock
def test_mcp_calls_pin_session_and_preserve_mcp_session_id() -> None:
    _mock_grant_and_revoke()
    seen_requests: list[httpx.Request] = []

    def handle_mcp(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json=_tool_envelope(body["id"], {"call": len(seen_requests)}),
            headers={"Mcp-Session-Id": "mcp_transport_1"},
        )

    route = respx.post(f"{BASE}/mcp").mock(side_effect=handle_mcp)

    with SimulationClient(api_key=ROOT_KEY, base_url=BASE) as client:
        with client.mcp_session(SESSION_ID) as mcp:
            assert mcp.call_tool("isaac.get_scene_info") == {"call": 1}
            assert mcp.call_tool("isaac.get_logs", {"lines": 20}) == {"call": 2}
            with pytest.raises(SimulationMCPError, match="does not match pinned session"):
                mcp.call_tool("isaac.get_scene_info", {"session_id": "sess_other"})
            with pytest.raises(SimulationMCPError, match=r"only permits isaac\.\*"):
                mcp.call_tool("sessions.list")

    assert route.call_count == 2
    first, second = seen_requests
    assert first.headers["Authorization"] == f"Bearer {SCOPED_KEY}"
    assert first.headers["X-Session-Id"] == SESSION_ID
    assert "Mcp-Session-Id" not in first.headers
    assert second.headers["Mcp-Session-Id"] == "mcp_transport_1"
    for request in seen_requests:
        body = json.loads(request.content)
        assert body["method"] == "tools/call"
        assert body["params"]["arguments"]["session_id"] == SESSION_ID


@respx.mock
def test_mcp_parses_tool_result_envelope() -> None:
    _mock_grant_and_revoke()
    expected = {"status": "success", "prim_count": 7}

    def handle_mcp(request: httpx.Request) -> httpx.Response:
        request_id = json.loads(request.content)["id"]
        return httpx.Response(200, json=_tool_envelope(request_id, expected))

    respx.post(f"{BASE}/mcp").mock(side_effect=handle_mcp)

    with SimulationClient(api_key=ROOT_KEY, base_url=BASE) as client:
        with client.mcp_session(SESSION_ID) as mcp:
            assert mcp.call_tool("isaac.get_scene_info") == expected


@pytest.mark.parametrize(
    ("mcp_response", "match"),
    [
        (httpx.Response(502), "HTTP 502"),
        (
            httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32603, "message": "gateway failed"},
                },
            ),
            "-32603: gateway failed",
        ),
        (
            httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "ok": False,
                                        "error": {
                                            "code": "BRIDGE_OFFLINE",
                                            "message": "Isaac bridge is offline",
                                        },
                                    }
                                ),
                            }
                        ]
                    },
                },
            ),
            "BRIDGE_OFFLINE: Isaac bridge is offline",
        ),
        (
            httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}}),
            "malformed envelope",
        ),
    ],
)
@respx.mock
def test_mcp_errors_are_diagnosable(
    mcp_response: httpx.Response,
    match: str,
) -> None:
    _mock_grant_and_revoke()
    respx.post(f"{BASE}/mcp").mock(return_value=mcp_response)

    with SimulationClient(api_key=ROOT_KEY, base_url=BASE) as client:
        with client.mcp_session(SESSION_ID) as mcp:
            with pytest.raises(SimulationMCPError, match=match):
                mcp.call_tool("isaac.get_scene_info")


@respx.mock
def test_mcp_error_redacts_scoped_key() -> None:
    _mock_grant_and_revoke()
    respx.post(f"{BASE}/mcp").mock(
        return_value=httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32603, "message": f"bad credential {SCOPED_KEY}"},
            },
        )
    )

    with SimulationClient(api_key=ROOT_KEY, base_url=BASE) as client:
        with client.mcp_session(SESSION_ID) as mcp:
            with pytest.raises(SimulationMCPError) as exc_info:
                mcp.call_tool("isaac.get_scene_info")

    assert SCOPED_KEY not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


@pytest.mark.parametrize(
    ("error_factory", "failure_code"),
    [
        (
            lambda request: httpx.ConnectError("connection failed", request=request),
            "MCP_TRANSPORT_CONNECT",
        ),
        (
            lambda request: httpx.ReadTimeout("request timed out", request=request),
            "MCP_TRANSPORT_TIMEOUT",
        ),
        (
            lambda request: httpx.RemoteProtocolError("protocol failed", request=request),
            "MCP_TRANSPORT_ERROR",
        ),
    ],
)
@respx.mock
def test_mcp_transport_error_hides_request_credentials(
    error_factory,
    failure_code: str,
) -> None:
    _mock_grant_and_revoke()

    def fail_with_request(request: httpx.Request) -> httpx.Response:
        raise error_factory(request)

    respx.post(f"{BASE}/mcp").mock(side_effect=fail_with_request)

    with SimulationClient(api_key=ROOT_KEY, base_url=BASE) as client:
        with client.mcp_session(SESSION_ID) as mcp:
            with pytest.raises(
                SimulationMCPError,
                match=rf"transport request failed \[{failure_code}\]",
            ) as exc_info:
                mcp.call_tool("isaac.get_scene_info")

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert SCOPED_KEY not in str(exc_info.value)


@respx.mock
def test_mcp_context_preserves_primary_error_when_revoke_is_unavailable() -> None:
    respx.post(f"{BASE}/v1/api-keys/session-scoped").mock(
        return_value=httpx.Response(201, json=_grant())
    )

    def fail_revoke(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("revocation unavailable", request=request)

    revoke = respx.delete(f"{BASE}/v1/api-keys/{KEY_ID}").mock(side_effect=fail_revoke)

    client = SimulationClient(api_key=ROOT_KEY, base_url=BASE)
    with pytest.raises(RuntimeError, match="primary failure") as exc_info:
        with client.mcp_session(SESSION_ID) as mcp:
            raise RuntimeError("primary failure")

    assert revoke.call_count == 1
    assert "closed=True" in repr(mcp)
    assert exc_info.value.__notes__ == [
        "SessionMCPClient cleanup failed after the primary error: "
        "ConnectError: revocation unavailable"
    ]

    with pytest.raises(httpx.ConnectError, match="revocation unavailable"):
        client.close()
    assert revoke.call_count == 2


@respx.mock
def test_simulation_context_preserves_primary_error_when_mcp_cleanup_fails() -> None:
    respx.post(f"{BASE}/v1/api-keys/session-scoped").mock(
        return_value=httpx.Response(201, json=_grant())
    )

    def fail_revoke(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("revocation unavailable", request=request)

    revoke = respx.delete(f"{BASE}/v1/api-keys/{KEY_ID}").mock(side_effect=fail_revoke)

    with pytest.raises(ValueError, match="primary failure") as exc_info:
        with SimulationClient(api_key=ROOT_KEY, base_url=BASE) as client:
            mcp = client.mcp_session(SESSION_ID)
            raise ValueError("primary failure")

    assert revoke.call_count == 1
    assert "closed=True" in repr(mcp)
    assert exc_info.value.__notes__ == [
        "SimulationClient cleanup failed after the primary error: "
        "ConnectError: revocation unavailable"
    ]


@respx.mock
def test_mcp_context_propagates_cleanup_error_without_primary_error() -> None:
    respx.post(f"{BASE}/v1/api-keys/session-scoped").mock(
        return_value=httpx.Response(201, json=_grant())
    )

    def fail_revoke(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("revocation unavailable", request=request)

    revoke = respx.delete(f"{BASE}/v1/api-keys/{KEY_ID}").mock(side_effect=fail_revoke)
    client = SimulationClient(api_key=ROOT_KEY, base_url=BASE)

    with pytest.raises(httpx.ConnectError, match="revocation unavailable"):
        with client.mcp_session(SESSION_ID):
            pass

    assert revoke.call_count == 1


@respx.mock
def test_simulation_context_propagates_cleanup_error_without_primary_error() -> None:
    respx.post(f"{BASE}/v1/api-keys/session-scoped").mock(
        return_value=httpx.Response(201, json=_grant())
    )

    def fail_revoke(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("revocation unavailable", request=request)

    revoke = respx.delete(f"{BASE}/v1/api-keys/{KEY_ID}").mock(side_effect=fail_revoke)

    with pytest.raises(httpx.ConnectError, match="revocation unavailable"):
        with SimulationClient(api_key=ROOT_KEY, base_url=BASE) as client:
            client.mcp_session(SESSION_ID)

    assert revoke.call_count == 1


@respx.mock
def test_failed_revocation_disables_calls_and_retries_until_confirmed() -> None:
    respx.post(f"{BASE}/v1/api-keys/session-scoped").mock(
        return_value=httpx.Response(201, json=_grant())
    )
    attempts = 0

    def revoke_once_unavailable(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("revocation unavailable", request=request)
        return httpx.Response(204)

    revoke = respx.delete(f"{BASE}/v1/api-keys/{KEY_ID}").mock(side_effect=revoke_once_unavailable)
    client = SimulationClient(api_key=ROOT_KEY, base_url=BASE)
    mcp = client.mcp_session(SESSION_ID)

    with pytest.raises(httpx.ConnectError, match="revocation unavailable"):
        mcp.close()
    with pytest.raises(SimulationMCPError, match="closed"):
        mcp.call_tool("isaac.get_scene_info")

    client.close()
    client.close()
    assert revoke.call_count == 2


@respx.mock
def test_simulation_client_context_revokes_key_without_stopping_session() -> None:
    _, revoke = _mock_grant_and_revoke()
    stop = respx.post(f"{BASE}/v1/sessions/{SESSION_ID}/stop").mock(
        return_value=httpx.Response(204)
    )

    with pytest.raises(RuntimeError, match="test body failed"):
        with SimulationClient(api_key=ROOT_KEY, base_url=BASE) as client:
            client.mcp_session(SESSION_ID)
            raise RuntimeError("test body failed")

    assert revoke.call_count == 1
    assert stop.call_count == 0
