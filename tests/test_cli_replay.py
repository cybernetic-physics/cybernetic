from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from cybernetics.cli.__main__ import main_cli
from cybernetics.replay import (
    AgentReplayBundle,
    ReplayEvent,
    ReplayEventSelection,
    ReplaySummary,
)


class FakeReplayClient:
    closed = 0

    def __init__(self, **kwargs: Any) -> None:
        pass

    def close(self) -> None:
        type(self).closed += 1

    def describe(self, session_id: str) -> ReplaySummary:
        return ReplaySummary.from_api_dict(
            {
                "sessionId": session_id,
                "formatVersion": 1,
                "recordings": [],
                "controlEvents": {"count": 0},
            }
        )

    def iter_events(self, session_id: str, query: Any):
        yield self._event()

    def select_events(self, session_id: str, query: Any) -> ReplayEventSelection:
        event = self._event()
        return ReplayEventSelection(
            session_id=session_id,
            start_time_ns=event.time_ns,
            end_time_ns=event.time_ns,
            query=query,
            events=(event,),
            matched_events=1,
            truncated=False,
            truncation_reason=None,
        )

    @staticmethod
    def _event() -> ReplayEvent:
        return ReplayEvent.from_api_dict(
            {
                "eventId": "evt_camera",
                "channel": "camera/main",
                "type": "camera.frame",
                "source": "isaac-sim",
                "timeNs": "1234567890123456789",
                "payload": {
                    "dataBase64": base64.b64encode(b"camera").decode("ascii"),
                    "access_token": "never-print-this",
                    "svg": "data:image/svg+xml,%3Csvg%3Eunsafe%3C/svg%3E",
                },
            }
        )

    def export_agent_bundle(self, session_id: str, destination: Path, **kwargs: Any):
        destination.mkdir(parents=True)
        paths = [
            destination / name
            for name in ("manifest.json", "context.md", "observations.ndjson", "events.ndjson")
        ]
        for path in paths:
            path.write_text("{}\n")
        return AgentReplayBundle(
            directory=destination,
            manifest_path=paths[0],
            context_path=paths[1],
            observations_path=paths[2],
            events_path=paths[3],
            frame_paths=(),
            file_sha256={},
            truncated=False,
            omissions=(),
            warnings=(),
        )


def test_top_level_help_lists_replay() -> None:
    result = CliRunner().invoke(main_cli, ["--help"])
    assert result.exit_code == 0
    assert "replay" in result.output


def test_replay_events_ndjson_never_prints_base64_or_secrets(monkeypatch) -> None:
    from cybernetics.cli.commands import replay

    FakeReplayClient.closed = 0
    monkeypatch.setattr(replay, "ReplayClient", FakeReplayClient)
    result = CliRunner().invoke(
        main_cli,
        ["replay", "events", "sess_demo", "--ndjson", "--channel", "camera/*"],
    )

    assert result.exit_code == 0, result.output
    event = json.loads(result.output)
    assert event["time_ns"] == "1234567890123456789"
    assert "dataBase64" not in result.output
    assert "never-print-this" not in result.output
    assert "data:image/svg+xml" not in result.output
    assert "[DATA URL OMITTED]" in result.output
    assert FakeReplayClient.closed == 1


def test_replay_events_ndjson_reports_truncation_on_stderr(monkeypatch) -> None:
    from cybernetics.cli.commands import replay

    class TruncatedReplayClient(FakeReplayClient):
        def select_events(self, session_id: str, query: Any) -> ReplayEventSelection:
            selection = super().select_events(session_id, query)
            return ReplayEventSelection(
                session_id=selection.session_id,
                start_time_ns=selection.start_time_ns,
                end_time_ns=selection.end_time_ns,
                query=selection.query,
                events=selection.events,
                matched_events=2,
                truncated=True,
                truncation_reason="max_events_before",
            )

    monkeypatch.setattr(replay, "ReplayClient", TruncatedReplayClient)
    result = CliRunner().invoke(
        main_cli,
        ["replay", "events", "sess_demo", "--ndjson", "--max-events", "1"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["event_id"] == "evt_camera"
    assert "2 matched, 1 returned (max_events_before)" in result.stderr


def test_replay_inspect_preserves_global_json_mode(monkeypatch) -> None:
    from cybernetics.cli.commands import replay

    FakeReplayClient.closed = 0
    monkeypatch.setattr(replay, "ReplayClient", FakeReplayClient)
    result = CliRunner().invoke(
        main_cli,
        ["--format", "json", "replay", "inspect", "sess_demo"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["session_id"] == "sess_demo"
    assert FakeReplayClient.closed == 1


def test_replay_export_closes_client_and_reports_bundle(monkeypatch, tmp_path: Path) -> None:
    from cybernetics.cli.commands import replay

    FakeReplayClient.closed = 0
    monkeypatch.setattr(replay, "ReplayClient", FakeReplayClient)
    output = tmp_path / "bundle"
    result = CliRunner().invoke(
        main_cli,
        ["--format", "json", "replay", "export", "sess_demo", "--out", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["directory"] == str(output)
    assert FakeReplayClient.closed == 1
