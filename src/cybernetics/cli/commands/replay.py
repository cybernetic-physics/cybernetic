"""Inspect and export durable session replay data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from cybernetics.replay import (
    AgentReplayOptions,
    ReplayClient,
    ReplayError,
    ReplayEventSelection,
    ReplayQuery,
)

from ..context import CLIContext
from ..exceptions import CyberneticsCliError
from ..output import OutputBase


class ReplayOutput(OutputBase):
    def __init__(self, title: str, payload: dict[str, Any]) -> None:
        self.title = title
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return self.payload

    def get_title(self) -> str | None:
        return self.title

    def get_table_columns(self) -> list[str]:
        return ["Field", "Value"]

    def get_table_rows(self) -> list[list[str]]:
        return [[key, _format_value(value)] for key, value in self.payload.items()]


class ReplayEventsOutput(OutputBase):
    def __init__(self, selection: ReplayEventSelection) -> None:
        self.selection = selection

    def to_dict(self) -> dict[str, Any]:
        return self.selection.to_dict()

    def get_title(self) -> str | None:
        suffix = " truncated" if self.selection.truncated else ""
        return (
            f"Replay Events ({len(self.selection.events)} of "
            f"{self.selection.matched_events}{suffix})"
        )

    def get_table_columns(self) -> list[str]:
        return ["Time (ns)", "Channel", "Type", "Source", "Event ID"]

    def get_table_rows(self) -> list[list[str]]:
        return [
            [str(event.time_ns), event.channel, event.type, event.source, event.event_id]
            for event in self.selection.events
        ]


@click.group()
def cli() -> None:
    """Inspect, stream, and export durable session replay data."""


@cli.command("inspect")
@click.argument("session_id")
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Control-plane API base URL.")
@click.pass_obj
def inspect_command(
    ctx: CLIContext | None,
    session_id: str,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Describe recordings, channels, time bounds, and control events."""
    client = _client(api_key=api_key, base_url=base_url)
    try:
        summary = client.describe(session_id)
    except ReplayError as exc:
        raise CyberneticsCliError(str(exc)) from exc
    finally:
        client.close()
    ReplayOutput("Session Replay", summary.to_dict()).print((ctx or CLIContext()).format)


@cli.command("events")
@click.argument("session_id")
@click.option("--channel", "channels", multiple=True, help="Exact channel or shell-style glob.")
@click.option("--recording", "recording_ids", multiple=True, help="Recording ID filter.")
@click.option("--start-time-ns", type=click.IntRange(min=0), default=None)
@click.option("--end-time-ns", type=click.IntRange(min=0), default=None)
@click.option(
    "--latest-seconds",
    type=click.FloatRange(min=0.001, max=300.0),
    default=30.0,
    show_default=True,
)
@click.option(
    "--max-events",
    type=click.IntRange(min=1, max=500),
    default=100,
    show_default=True,
)
@click.option(
    "--control-events/--no-control-events",
    default=True,
    show_default=True,
    help="Include control-plane session events.",
)
@click.option("--ndjson", is_flag=True, help="Write one text-safe event JSON object per line.")
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Control-plane API base URL.")
@click.pass_obj
def events_command(
    ctx: CLIContext | None,
    session_id: str,
    channels: tuple[str, ...],
    recording_ids: tuple[str, ...],
    start_time_ns: int | None,
    end_time_ns: int | None,
    latest_seconds: float,
    max_events: int,
    control_events: bool,
    ndjson: bool,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Read a bounded, time-ordered raw event window."""
    try:
        query = ReplayQuery(
            channels=channels,
            recording_ids=recording_ids,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            latest_seconds=latest_seconds,
            max_events=max_events,
            max_images=0,
            include_control_events=control_events,
        )
        client = _client(api_key=api_key, base_url=base_url)
        try:
            selection = client.select_events(session_id, query)
        finally:
            client.close()
    except ReplayError as exc:
        raise CyberneticsCliError(str(exc)) from exc

    if ndjson:
        for event in selection.events:
            click.echo(json.dumps(event.to_dict(), separators=(",", ":"), sort_keys=True))
        if selection.truncated:
            click.echo(
                "Replay selection truncated: "
                f"{selection.matched_events} matched, {len(selection.events)} returned "
                f"({selection.truncation_reason}). Use a narrower window or channel filter.",
                err=True,
            )
        return
    ReplayEventsOutput(selection).print((ctx or CLIContext()).format)


@cli.command("export")
@click.argument("session_id")
@click.option(
    "--out",
    "destination",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Destination directory for the agent bundle.",
)
@click.option("--profile", type=click.Choice(["agent"]), default="agent", show_default=True)
@click.option("--channel", "channels", multiple=True, help="Exact channel or shell-style glob.")
@click.option("--start-time-ns", type=click.IntRange(min=0), default=None)
@click.option("--end-time-ns", type=click.IntRange(min=0), default=None)
@click.option(
    "--latest-seconds",
    type=click.FloatRange(min=0.001, max=300.0),
    default=30.0,
    show_default=True,
)
@click.option("--max-events", type=click.IntRange(min=1, max=500), default=100, show_default=True)
@click.option("--max-images", type=click.IntRange(min=0, max=8), default=4, show_default=True)
@click.option("--overwrite", is_flag=True, help="Replace a prior managed replay bundle.")
@click.option("--api-key", default=None, help="API key override.")
@click.option("--base-url", default=None, help="Control-plane API base URL.")
@click.pass_obj
def export_command(
    ctx: CLIContext | None,
    session_id: str,
    destination: Path,
    profile: str,
    channels: tuple[str, ...],
    start_time_ns: int | None,
    end_time_ns: int | None,
    latest_seconds: float,
    max_events: int,
    max_images: int,
    overwrite: bool,
    api_key: str | None,
    base_url: str | None,
) -> None:
    """Export a deterministic text-plus-image bundle for multimodal agents."""
    try:
        query = ReplayQuery(
            channels=channels,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            latest_seconds=latest_seconds,
            max_events=max_events,
            max_images=max_images,
        )
        options = AgentReplayOptions(
            max_events=max_events,
            max_images=max_images,
            overwrite=overwrite,
            profile=profile,  # type: ignore[arg-type]
        )
        client = _client(api_key=api_key, base_url=base_url)
        try:
            bundle = client.export_agent_bundle(
                session_id,
                destination,
                query=query,
                options=options,
            )
        finally:
            client.close()
    except ReplayError as exc:
        raise CyberneticsCliError(str(exc)) from exc
    ReplayOutput("Agent Replay Bundle", bundle.to_dict()).print((ctx or CLIContext()).format)


def _client(*, api_key: str | None, base_url: str | None) -> ReplayClient:
    try:
        return ReplayClient(api_key=api_key, base_url=base_url)
    except ReplayError as exc:
        raise CyberneticsCliError(str(exc)) from exc


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return "" if value is None else str(value)
