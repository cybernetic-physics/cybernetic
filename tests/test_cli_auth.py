"""Contract tests for the ``cybernetics auth`` group: the device-poll state
machine (pure, injectable clock + fake transport) and the simple subcommands.

The full browser round-trip needs a live control plane and is exercised
end-to-end against the hosted API; here we pin the poll logic + key resolution.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from cybernetics.cli.commands import auth
from cybernetics.cli.exceptions import CyberneticsCliError


class _FakeResponse:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict:
        return self._body


class _FakeHttp:
    """A canned /v1/auth/device/token transport returning a scripted sequence."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def post(self, path: str, json: dict) -> _FakeResponse:  # noqa: A002
        self.calls += 1
        return self._responses.pop(0)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for var in ("CYBERNETICS_API_KEY", "WORLDLINES_API_KEY", "CYBERNETICS_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_auth_group_help_lists_subcommands() -> None:
    result = CliRunner().invoke(auth.cli, ["--help"])
    assert result.exit_code == 0
    for sub in ("login", "logout", "status", "token"):
        assert sub in result.output


def test_token_prints_env_key() -> None:
    runner = CliRunner()
    result = runner.invoke(auth.cli, ["token"], env={"CYBERNETICS_API_KEY": "cp_live_xyz"})
    assert result.exit_code == 0
    assert result.output.strip() == "cp_live_xyz"


def test_token_without_key_errors() -> None:
    result = CliRunner().invoke(auth.cli, ["token"])
    assert result.exit_code != 0


def test_poll_pending_then_approved() -> None:
    clock = _Clock()
    http = _FakeHttp(
        [
            _FakeResponse(400, {"error": "authorization_pending"}),
            _FakeResponse(200, {"access_token": "cp_live_ok", "api_key_id": "key_1"}),
        ]
    )
    body = auth._poll_for_token(
        http,
        device_code="cp_dev_x",
        interval=5,
        expires_in=900,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert body["access_token"] == "cp_live_ok"
    assert http.calls == 2


def test_poll_slow_down_increases_wait() -> None:
    clock = _Clock()
    http = _FakeHttp(
        [
            _FakeResponse(400, {"error": "slow_down"}),
            _FakeResponse(200, {"access_token": "cp_live_ok"}),
        ]
    )
    auth._poll_for_token(
        http,
        device_code="d",
        interval=5,
        expires_in=900,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    # first sleep at the base interval, second sleep bumped by +5 after slow_down
    assert clock.sleeps == [5, 10]


def test_poll_expired_raises() -> None:
    clock = _Clock()
    http = _FakeHttp([_FakeResponse(400, {"error": "expired_token"})])
    with pytest.raises(CyberneticsCliError, match="expired"):
        auth._poll_for_token(
            http,
            device_code="d",
            interval=5,
            expires_in=900,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
