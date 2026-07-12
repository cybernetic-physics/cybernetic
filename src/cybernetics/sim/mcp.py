"""Session-pinned MCP access for hosted Isaac simulations."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from .errors import SimulationMCPError


class SessionMCPClient:
    """Call ``isaac.*`` tools with a private, session-scoped credential."""

    __slots__ = (
        "_base_url",
        "_client",
        "_closed",
        "_key_id",
        "_mcp_session_id",
        "_request_id",
        "_revoke_key",
        "_scoped_key",
        "_session_id",
    )

    def __init__(
        self,
        *,
        base_url: str,
        http_client: Any,
        session_id: str,
        key_id: str,
        scoped_key: str,
        revoke_key: Callable[[str], None],
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = http_client
        self._session_id = session_id
        self._key_id = key_id
        self._scoped_key = scoped_key
        self._revoke_key = revoke_key
        self._mcp_session_id: str | None = None
        self._request_id = 0
        self._closed = False

    @property
    def session_id(self) -> str:
        """The hosted session every tool call is pinned to."""
        return self._session_id

    def __repr__(self) -> str:
        return f"SessionMCPClient(session_id={self._session_id!r}, closed={self._closed!r})"

    def __enter__(self) -> "SessionMCPClient":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Revoke the scoped key without changing the hosted session lifecycle."""
        if self._closed:
            return
        self._revoke_key(self._key_id)
        self._closed = True
        self._scoped_key = ""

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call one ``isaac.*`` tool against this client's hosted session."""
        self._ensure_open()
        if not isinstance(name, str) or not name.startswith("isaac.") or name == "isaac.":
            raise SimulationMCPError("session-scoped simulation MCP only permits isaac.* tools")
        if arguments is not None and not isinstance(arguments, dict):
            raise SimulationMCPError("MCP tool arguments must be an object")

        tool_arguments = dict(arguments or {})
        requested_session = tool_arguments.get("session_id")
        if requested_session is not None and requested_session != self._session_id:
            raise SimulationMCPError(
                f"MCP tool target {requested_session!r} does not match pinned session "
                f"{self._session_id!r}"
            )
        tool_arguments["session_id"] = self._session_id

        self._request_id += 1
        request_id = self._request_id
        headers = {
            "Authorization": f"Bearer {self._scoped_key}",
            "Accept": "application/json",
            "X-Session-Id": self._session_id,
        }
        if self._mcp_session_id is not None:
            headers["Mcp-Session-Id"] = self._mcp_session_id

        response = self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": tool_arguments},
            },
            headers,
        )
        if response is None:
            raise SimulationMCPError(f"MCP tool {name!r} transport request failed")
        status_code = getattr(response, "status_code", 0)
        if status_code >= 400:
            raise SimulationMCPError(f"MCP tool {name!r} failed with HTTP {status_code}")

        self._preserve_mcp_session(response)
        try:
            envelope = response.json()
        except (TypeError, ValueError) as exc:
            raise SimulationMCPError(f"MCP tool {name!r} returned invalid JSON") from exc
        return self._parse_envelope(name, request_id, envelope)

    def _send(self, payload: dict[str, Any], headers: dict[str, str]) -> Any | None:
        try:
            return self._client.request(
                "POST",
                f"{self._base_url}/mcp",
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError:
            return None

    def _preserve_mcp_session(self, response: Any) -> None:
        returned_id = response.headers.get("Mcp-Session-Id")
        if not returned_id:
            return
        if self._mcp_session_id is None:
            self._mcp_session_id = returned_id
            return
        if returned_id != self._mcp_session_id:
            raise SimulationMCPError("MCP gateway changed the established Mcp-Session-Id")

    def _parse_envelope(
        self,
        name: str,
        request_id: int,
        envelope: Any,
    ) -> dict[str, Any]:
        if not isinstance(envelope, dict):
            raise self._malformed(name, "response is not an object")
        if envelope.get("jsonrpc") != "2.0" or envelope.get("id") != request_id:
            raise self._malformed(name, "JSON-RPC version or request id does not match")

        rpc_error = envelope.get("error")
        if rpc_error:
            raise SimulationMCPError(f"MCP tool {name!r} failed: {self._error_summary(rpc_error)}")

        result = envelope.get("result")
        if not isinstance(result, dict):
            raise self._malformed(name, "result is missing or is not an object")
        content = result.get("content")
        if not isinstance(content, list):
            raise self._malformed(name, "result.content is not an array")
        text = next(
            (
                item.get("text")
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ),
            None,
        )
        if text is None:
            raise self._malformed(name, "result.content has no text payload")
        try:
            payload = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise self._malformed(name, "tool text payload is not JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("ok"), bool):
            raise self._malformed(name, "tool payload is missing a boolean ok field")

        if payload["ok"] is False:
            raise SimulationMCPError(
                f"MCP tool {name!r} failed: {self._error_summary(payload.get('error'))}"
            )
        if result.get("isError") is True:
            raise SimulationMCPError(f"MCP tool {name!r} reported an MCP tool error")

        data = payload.get("data")
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise self._malformed(name, "tool payload data is not an object")
        return data

    def _error_summary(self, error: Any) -> str:
        if not isinstance(error, dict):
            return "unknown error"
        code = error.get("code", "UNKNOWN")
        message = error.get("message", "unknown error")
        summary = f"{code}: {message}"
        if self._scoped_key:
            summary = summary.replace(self._scoped_key, "[REDACTED]")
        return summary[:500]

    @staticmethod
    def _malformed(name: str, detail: str) -> SimulationMCPError:
        return SimulationMCPError(f"MCP tool {name!r} returned a malformed envelope: {detail}")

    def _ensure_open(self) -> None:
        if self._closed:
            raise SimulationMCPError("session-scoped MCP client is closed")
