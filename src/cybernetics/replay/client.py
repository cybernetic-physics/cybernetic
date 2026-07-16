"""Authenticated synchronous client for durable session replay."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import quote

from cybernetics.lib.credentials import resolve_api_key, resolve_base_url

from .errors import ReplayError, ReplayIntegrityError, ReplaySchemaError
from .models import (
    _MAX_TIME_NS,
    AgentReplayBundle,
    AgentReplayOptions,
    ReplayChunk,
    ReplayEvent,
    ReplayEventSelection,
    ReplayObservation,
    ReplayQuery,
    ReplaySummary,
)

DEFAULT_BASE_URL = "https://api.cyberneticphysics.com"
_MAX_COMPRESSED_CHUNK_BYTES = 32 * 1024 * 1024
_MAX_DECODED_CHUNK_BYTES = 128 * 1024 * 1024
_MAX_EVENT_LINE_BYTES = 8 * 1024 * 1024
_MAX_SELECTED_CHUNKS = 128
_MAX_SELECTED_COMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_DECODED_BYTES = 256 * 1024 * 1024
_MAX_EVENTS_SCANNED = 100_000
_MAX_CHUNK_INDEX_PAGES = 64
_MAX_CHUNKS_INDEXED = 10_000
_MAX_CONTROL_EVENT_PAGES = 64
_MAX_CONTROL_EVENTS_SCANNED = 10_000
_MAX_JSON_RESPONSE_BYTES = 192 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_DEFAULT_WINDOW_NS = 30_000_000_000
_TIMELINE_CONTENT_TYPES = {
    "application/vnd.cybernetic.timeline+ndjson",
    "application/x-ndjson",
}


@dataclass
class _ChunkIndexBudget:
    pages: int = 0
    chunks: int = 0


_RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


class ReplayClient:
    """Read immutable replay data and project bounded agent observations."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: Any = None,
    ) -> None:
        resolved_key = resolve_api_key(api_key)
        if not resolved_key:
            raise ReplayError(
                "No API key found. Run 'cybernetics auth login' or set CYBERNETICS_API_KEY."
            )
        self.api_key = resolved_key
        self.base_url = (resolve_base_url(base_url) or DEFAULT_BASE_URL).rstrip("/")
        self._owns_client = http_client is None
        if http_client is None:
            import httpx

            http_client = httpx.Client(base_url=self.base_url, timeout=180.0)
        self._client = http_client

    def close(self) -> None:
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "ReplayClient":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self.close()
        except Exception as close_error:
            if isinstance(exc, BaseException):
                exc.add_note(
                    "Replay client cleanup failed while handling the primary error: "
                    f"{type(close_error).__name__}: {close_error}"
                )
                return
            raise

    def describe(self, session_id: str) -> ReplaySummary:
        """Return the bounded timeline index and discovered channel catalog."""

        return self.get_summary(session_id)

    def get_summary(self, session_id: str) -> ReplaySummary:
        session_id = _resource_id(session_id, "session_id")
        body = self._request_json(
            "GET", f"/v1/sessions/{_path_segment(session_id, 'session_id')}/timeline"
        )
        summary = ReplaySummary.from_api_dict(body)
        if summary.session_id != session_id:
            raise ReplayIntegrityError(
                "replay summary session_id does not match the requested session"
            )
        return summary

    def iter_events(
        self,
        session_id: str,
        query: ReplayQuery | None = None,
    ) -> Iterator[ReplayEvent]:
        """Yield a bounded, deterministic raw event selection.

        Raw reads use the existing index/content/control APIs. An explicit
        window returns its earliest ``max_events``; the implicit latest window
        returns its latest ``max_events``. Event objects retain camera bytes for
        programmatic use, while their default ``to_dict()`` representation
        omits embedded base64.
        """

        selection = self.select_events(session_id, query)
        yield from selection.events

    def select_events(
        self,
        session_id: str,
        query: ReplayQuery | None = None,
    ) -> ReplayEventSelection:
        """Return raw events with explicit match and truncation metadata."""

        return self._collect_events(session_id, query or ReplayQuery())

    def get_observation(
        self,
        session_id: str,
        query: ReplayQuery | None = None,
        *,
        image_data: bool = False,
    ) -> ReplayObservation:
        """Return one server-projected multimodal replay window."""

        session_id = _resource_id(session_id, "session_id")
        if not isinstance(image_data, bool):
            raise ReplaySchemaError("image_data must be a boolean")
        requested = query or ReplayQuery()
        _validate_observation_query(requested)
        selected = self._resolve_observation_query(requested)
        implicit_latest = selected.start_time_ns is None and selected.end_time_ns is None
        if implicit_latest and selected.latest_seconds != 30.0:
            anchor_query = replace(selected, max_events=1, max_images=0)
            anchor = self._request_observation(
                session_id,
                anchor_query,
                image_data=False,
            )
            selected = _anchor_latest_query(selected, anchor.end_time_ns)
        observation = self._request_observation(
            session_id,
            selected,
            image_data=image_data,
        )
        if implicit_latest and selected.start_time_ns is None and selected.end_time_ns is None:
            selected = replace(
                selected,
                start_time_ns=observation.start_time_ns,
                end_time_ns=observation.end_time_ns,
                latest_seconds=None,
            )
        elif (
            observation.start_time_ns != selected.start_time_ns
            or observation.end_time_ns != selected.end_time_ns
        ):
            raise ReplayIntegrityError(
                "replay observation window does not match the resolved query window"
            )
        return replace(observation, effective_query=selected)

    def query(
        self,
        session_id: str,
        query: ReplayQuery | None = None,
        *,
        image_data: bool = False,
    ) -> ReplayObservation:
        """Alias for :meth:`get_observation` for agent query loops."""

        return self.get_observation(session_id, query, image_data=image_data)

    def export_agent_bundle(
        self,
        session_id: str,
        destination: str | Path,
        *,
        query: ReplayQuery | None = None,
        options: AgentReplayOptions | None = None,
    ) -> AgentReplayBundle:
        """Write a deterministic, image-separated agent replay bundle."""

        from .agent import export_agent_bundle

        return export_agent_bundle(
            self,
            session_id,
            destination,
            query=query,
            options=options,
        )

    def _collect_events(
        self,
        session_id: str,
        query: ReplayQuery,
    ) -> ReplayEventSelection:
        session_id = _resource_id(session_id, "session_id")
        summary = self.get_summary(session_id)
        start_time_ns, end_time_ns = _resolve_time_bounds(summary, query)
        recording_filter = set(query.recording_ids)

        events: list[ReplayEvent] = []
        selected_chunks: list[ReplayChunk] = []
        selected_compressed_bytes = 0
        seen_chunks: set[tuple[str, str]] = set()
        recordings = [
            recording
            for recording in summary.recordings
            if not recording_filter or recording.recording_id in recording_filter
        ]
        if len(recordings) > 256:
            raise ReplayError(
                "raw replay selection exceeds the 256-recording scan budget; use a recording filter"
            )
        index_budget = _ChunkIndexBudget()
        for recording in recordings:
            for chunk in self._iter_chunks(
                session_id,
                recording.recording_id,
                budget=index_budget,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
            ):
                if (
                    recording.start_time_ns is not None
                    and chunk.start_time_ns < recording.start_time_ns
                ) or (
                    recording.end_time_ns is not None
                    and chunk.end_time_ns > recording.end_time_ns
                ):
                    raise ReplayIntegrityError(
                        f"chunk {chunk.chunk_id} falls outside recording "
                        f"{recording.recording_id} time bounds"
                    )
                chunk_identity = (chunk.recording_id, chunk.chunk_id)
                if chunk_identity in seen_chunks:
                    continue
                seen_chunks.add(chunk_identity)
                if not chunk.overlaps(start_time_ns, end_time_ns):
                    continue
                if query.channels and not any(
                    _channel_matches(channel, query.channels) for channel in chunk.channels
                ):
                    continue
                selected_chunks.append(chunk)
                selected_compressed_bytes += chunk.size_bytes
                if (
                    len(selected_chunks) > _MAX_SELECTED_CHUNKS
                    or selected_compressed_bytes > _MAX_SELECTED_COMPRESSED_BYTES
                ):
                    raise ReplayError(
                        "raw replay selection exceeds the 128 chunk / 256 MiB scan budget; "
                        "use a narrower time window, channel filter, or recording filter"
                    )

        decoded_bytes_scanned = 0
        events_scanned = 0
        for chunk in selected_chunks:
            if events_scanned + chunk.event_count > _MAX_EVENTS_SCANNED:
                raise ReplayError(
                    "raw replay selection exceeds the 100000 event scan budget; "
                    "use a narrower time window, channel filter, or recording filter"
                )
            chunk_events, decoded_bytes = self._download_chunk_events(
                session_id,
                chunk,
                max_decoded_bytes=_MAX_TOTAL_DECODED_BYTES - decoded_bytes_scanned,
            )
            decoded_bytes_scanned += decoded_bytes
            events_scanned += len(chunk_events)
            if (
                decoded_bytes_scanned > _MAX_TOTAL_DECODED_BYTES
                or events_scanned > _MAX_EVENTS_SCANNED
            ):
                raise ReplayError(
                    "raw replay selection exceeds the 256 MiB / 100000 event decode budget; "
                    "use a narrower time window, channel filter, or recording filter"
                )
            for event in chunk_events:
                if _event_matches(event, query.channels, start_time_ns, end_time_ns):
                    events.append(event)

        if query.include_control_events:
            for event in self._iter_control_events(session_id):
                if _event_matches(event, query.channels, start_time_ns, end_time_ns):
                    events.append(event)

        ordered = sorted(
            events,
            key=lambda event: (
                event.time_ns,
                event.source,
                event.event_id,
            ),
        )
        matched_events = len(ordered)
        truncated = matched_events > query.max_events
        latest_selection = query.start_time_ns is None and query.end_time_ns is None
        if truncated and latest_selection:
            ordered = ordered[-query.max_events :]
        elif truncated:
            ordered = ordered[: query.max_events]
        return ReplayEventSelection(
            session_id=session_id,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            query=replace(
                query,
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                latest_seconds=None,
            ),
            events=tuple(ordered),
            matched_events=matched_events,
            truncated=truncated,
            truncation_reason=(
                "max_events_before" if latest_selection else "max_events_after"
            )
            if truncated
            else None,
        )

    def _iter_chunks(
        self,
        session_id: str,
        recording_id: str,
        *,
        budget: _ChunkIndexBudget | None = None,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
    ) -> Iterator[ReplayChunk]:
        session_id = _resource_id(session_id, "session_id")
        recording_id = _resource_id(recording_id, "recording_id")
        if start_time_ns is not None:
            _query_time_ns(start_time_ns, "start_time_ns")
        if end_time_ns is not None:
            _query_time_ns(end_time_ns, "end_time_ns")
        if (
            start_time_ns is not None
            and end_time_ns is not None
            and end_time_ns < start_time_ns
        ):
            raise ReplaySchemaError("chunk query end_time_ns must not precede start_time_ns")
        index_budget = budget or _ChunkIndexBudget()
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            index_budget.pages += 1
            if index_budget.pages > _MAX_CHUNK_INDEX_PAGES:
                raise ReplayError(
                    "raw replay chunk index exceeds 64 pages; use a narrower recording"
                )
            params: list[tuple[str, str]] = [("limit", "500")]
            if start_time_ns is not None:
                params.append(("startTimeNs", str(start_time_ns)))
            if end_time_ns is not None:
                params.append(("endTimeNs", str(end_time_ns)))
            if cursor:
                params.append(("cursor", cursor))
            body = self._request_json(
                "GET",
                "/v1/sessions/"
                f"{_path_segment(session_id, 'session_id')}/timeline/recordings/"
                f"{_path_segment(recording_id, 'recording_id')}/chunks",
                params=params,
            )
            items = body.get("items", [])
            if not isinstance(items, list):
                raise ReplaySchemaError("timeline chunk page items must be an array")
            index_budget.chunks += len(items)
            if index_budget.chunks > _MAX_CHUNKS_INDEXED:
                raise ReplayError(
                    "raw replay chunk index exceeds 10000 entries; use a narrower recording"
                )
            for item in items:
                if not isinstance(item, Mapping):
                    raise ReplaySchemaError("timeline chunk page item must be an object")
                chunk = ReplayChunk.from_api_dict(item)
                if chunk.recording_id != recording_id:
                    raise ReplayIntegrityError(
                        "timeline chunk recording_id does not match the requested recording"
                    )
                yield chunk
            cursor = _optional_page_string(body.get("nextCursor"), "nextCursor")
            if cursor is None:
                return
            if cursor in seen_cursors:
                raise ReplaySchemaError("timeline chunk pagination repeated a cursor")
            seen_cursors.add(cursor)

    def _iter_control_events(self, session_id: str) -> Iterator[ReplayEvent]:
        after_id: str | None = None
        seen_after_ids: set[str] = set()
        scanned = 0
        pages = 0
        while True:
            pages += 1
            if pages > _MAX_CONTROL_EVENT_PAGES:
                raise ReplayError(
                    "raw replay control-event index exceeds 64 pages; "
                    "set include_control_events=False or use the observation API"
                )
            params: list[tuple[str, str]] = [("limit", "1000")]
            if after_id:
                params.append(("afterId", after_id))
            body = self._request_json(
                "GET",
                f"/v1/sessions/{_path_segment(session_id, 'session_id')}/timeline/control-events",
                params=params,
            )
            items = body.get("items", [])
            if not isinstance(items, list):
                raise ReplaySchemaError("timeline control page items must be an array")
            for item in items:
                if not isinstance(item, Mapping):
                    raise ReplaySchemaError("timeline control event must be an object")
                scanned += 1
                if scanned > _MAX_CONTROL_EVENTS_SCANNED:
                    raise ReplayError(
                        "raw replay control-event scan exceeds 10000 events; "
                        "set include_control_events=False or use the observation API"
                    )
                yield ReplayEvent.from_api_dict(item)
            after_id = _optional_page_string(body.get("nextAfterId"), "nextAfterId")
            if after_id is None:
                return
            if after_id in seen_after_ids:
                raise ReplaySchemaError("timeline control pagination repeated afterId")
            seen_after_ids.add(after_id)

    def _download_chunk_events(
        self,
        session_id: str,
        chunk: ReplayChunk,
        *,
        max_decoded_bytes: int,
    ) -> tuple[tuple[ReplayEvent, ...], int]:
        if chunk.compression != "gzip":
            raise ReplayIntegrityError(
                f"chunk {chunk.chunk_id} uses unsupported compression {chunk.compression!r}"
            )
        if chunk.content_type.split(";", 1)[0].lower() not in _TIMELINE_CONTENT_TYPES:
            raise ReplayIntegrityError(
                f"chunk {chunk.chunk_id} uses unsupported content type {chunk.content_type!r}"
            )
        if chunk.size_bytes > _MAX_COMPRESSED_CHUNK_BYTES:
            raise ReplayIntegrityError(
                f"chunk {chunk.chunk_id} exceeds the 32 MiB compressed replay limit"
            )
        path = (
            f"/v1/sessions/{_path_segment(session_id, 'session_id')}/timeline/recordings/"
            f"{_path_segment(chunk.recording_id, 'recording_id')}/chunks/"
            f"{_path_segment(chunk.chunk_id, 'chunk_id')}/content"
        )
        operation = f"GET {path}"
        try:
            with self._client.stream("GET", path, headers=self._auth_headers()) as response:
                self._raise_for_response(response, operation)
                parts: list[bytes] = []
                size = 0
                for part in response.iter_raw():
                    size += len(part)
                    if size > _MAX_COMPRESSED_CHUNK_BYTES:
                        raise ReplayIntegrityError(
                            f"chunk {chunk.chunk_id} exceeded the compressed replay limit"
                        )
                    parts.append(part)
                raw = b"".join(parts)
        except ReplayError:
            raise
        except Exception as exc:
            raise ReplayError(f"{operation} failed: {type(exc).__name__}") from exc

        if not raw.startswith(b"\x1f\x8b"):
            raise ReplayIntegrityError(f"chunk {chunk.chunk_id} is not gzip data")
        if len(raw) != chunk.size_bytes:
            raise ReplayIntegrityError(
                f"chunk {chunk.chunk_id} compressed size does not match its index"
            )
        digest = hashlib.sha256(raw).hexdigest()
        if digest != chunk.sha256:
            raise ReplayIntegrityError(f"chunk {chunk.chunk_id} SHA-256 does not match its index")
        text, decoded_size = _bounded_gzip_decode(
            raw,
            chunk.chunk_id,
            max_decoded_bytes=max_decoded_bytes,
        )
        events = _parse_ndjson_events(text, chunk.chunk_id)
        if len(events) != chunk.event_count:
            raise ReplayIntegrityError(
                f"chunk {chunk.chunk_id} event count does not match its index"
            )
        for event in events:
            if event.time_ns < chunk.start_time_ns or event.time_ns > chunk.end_time_ns:
                raise ReplayIntegrityError(
                    f"chunk {chunk.chunk_id} contains an event outside its indexed time bounds"
                )
            if chunk.channels and event.channel not in chunk.channels:
                raise ReplayIntegrityError(
                    f"chunk {chunk.chunk_id} contains an event outside its channel index"
                )
        return events, decoded_size

    def _resolve_observation_query(
        self,
        query: ReplayQuery,
    ) -> ReplayQuery:
        if query.start_time_ns is None and query.end_time_ns is None:
            return query
        if query.start_time_ns is not None and query.end_time_ns is not None:
            return replace(query, latest_seconds=None)
        start_time_ns, end_time_ns = _resolve_partial_time_bounds(query)
        return replace(
            query,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            latest_seconds=None,
        )

    def _request_observation(
        self,
        session_id: str,
        query: ReplayQuery,
        *,
        image_data: bool,
    ) -> ReplayObservation:
        body = self._request_json(
            "GET",
            f"/v1/sessions/{_path_segment(session_id, 'session_id')}/timeline/observations",
            params=query.to_api_params(image_data="inline" if image_data else "omit"),
        )
        observation = ReplayObservation.from_api_dict(body)
        if observation.session_id != session_id:
            raise ReplayIntegrityError(
                "replay observation session_id does not match the requested session"
            )
        _validate_observation_response(observation, query, image_data=image_data)
        return observation

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        operation = f"{method} {path}"
        try:
            response = self._client.request(
                method,
                path,
                params=params,
                headers=self._auth_headers(),
            )
        except Exception as exc:
            raise ReplayError(f"{operation} failed: {type(exc).__name__}") from exc
        self._raise_for_response(response, operation)
        try:
            raw = response.content
        except Exception as exc:
            raise ReplayError(f"{operation} response body could not be read") from exc
        if not isinstance(raw, bytes):
            raise ReplaySchemaError(f"{operation} returned a non-byte response body")
        if len(raw) > _MAX_JSON_RESPONSE_BYTES:
            raise ReplaySchemaError(f"{operation} returned JSON above the replay response limit")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReplaySchemaError(f"{operation} returned invalid UTF-8 JSON") from exc
        body = _strict_json_loads(text, operation)
        if not isinstance(body, dict):
            raise ReplaySchemaError(f"{operation} returned a non-object JSON payload")
        return body

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _raise_for_response(response: Any, operation: str) -> None:
        status_code = int(getattr(response, "status_code", 0))
        if status_code < 400:
            return
        raise ReplayError(f"{operation} failed with HTTP {status_code}")


def _bounded_gzip_decode(
    raw: bytes,
    chunk_id: str,
    *,
    max_decoded_bytes: int,
) -> tuple[str, int]:
    allowed_bytes = min(_MAX_DECODED_CHUNK_BYTES, max_decoded_bytes)
    if allowed_bytes < 1:
        raise ReplayIntegrityError("raw replay decoded-byte budget is exhausted")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as archive:
            decoded = archive.read(allowed_bytes + 1)
    except (OSError, EOFError) as exc:
        raise ReplayIntegrityError(f"chunk {chunk_id} contains invalid gzip data") from exc
    if len(decoded) > allowed_bytes:
        raise ReplayIntegrityError(f"chunk {chunk_id} exceeded the remaining decoded replay budget")
    try:
        return decoded.decode("utf-8"), len(decoded)
    except UnicodeDecodeError as exc:
        raise ReplayIntegrityError(f"chunk {chunk_id} decoded to invalid UTF-8") from exc


def _parse_ndjson_events(text: str, chunk_id: str) -> tuple[ReplayEvent, ...]:
    events: list[ReplayEvent] = []
    for line_number, line in enumerate(io.StringIO(text), start=1):
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > _MAX_EVENT_LINE_BYTES:
            raise ReplayIntegrityError(
                f"chunk {chunk_id} line {line_number} exceeds the event line limit"
            )
        item = _strict_json_loads(line, f"chunk {chunk_id} line {line_number}")
        if not isinstance(item, Mapping):
            raise ReplaySchemaError(f"chunk {chunk_id} line {line_number} is not an object")
        events.append(ReplayEvent.from_api_dict(item))
    return tuple(events)


def _resolve_time_bounds(summary: ReplaySummary, query: ReplayQuery) -> tuple[int, int]:
    start_time_ns = query.start_time_ns
    end_time_ns = query.end_time_ns
    if start_time_ns is not None and end_time_ns is not None:
        return start_time_ns, end_time_ns
    duration_ns = (
        int(query.latest_seconds * 1_000_000_000)
        if query.latest_seconds is not None
        else _DEFAULT_WINDOW_NS
    )
    if start_time_ns is not None:
        end_time_ns = start_time_ns + duration_ns
        if end_time_ns > _MAX_TIME_NS:
            raise ReplaySchemaError(
                "resolved replay end_time_ns exceeds PostgreSQL's signed BIGINT limit"
            )
        return start_time_ns, end_time_ns
    if end_time_ns is not None:
        return max(0, end_time_ns - duration_ns), end_time_ns
    endings = [
        recording.end_time_ns
        for recording in summary.recordings
        if recording.end_time_ns is not None
    ]
    if summary.control_last_time_ns is not None:
        endings.append(summary.control_last_time_ns)
    end_time_ns = max(endings) if endings else 0
    start_time_ns = max(0, end_time_ns - duration_ns)
    return start_time_ns, end_time_ns


def _resolve_partial_time_bounds(query: ReplayQuery) -> tuple[int, int]:
    duration_ns = int((query.latest_seconds or 30.0) * 1_000_000_000)
    if query.start_time_ns is not None:
        end_time_ns = query.start_time_ns + duration_ns
        if end_time_ns > _MAX_TIME_NS:
            raise ReplaySchemaError(
                "resolved replay end_time_ns exceeds PostgreSQL's signed BIGINT limit"
            )
        return query.start_time_ns, end_time_ns
    if query.end_time_ns is not None:
        return max(0, query.end_time_ns - duration_ns), query.end_time_ns
    raise ReplaySchemaError("partial replay bounds require one explicit timestamp")


def _anchor_latest_query(query: ReplayQuery, end_time_ns: int) -> ReplayQuery:
    duration_ns = int((query.latest_seconds or 30.0) * 1_000_000_000)
    return replace(
        query,
        start_time_ns=max(0, end_time_ns - duration_ns),
        end_time_ns=end_time_ns,
        latest_seconds=None,
    )


def _event_matches(
    event: ReplayEvent,
    channels: tuple[str, ...],
    start_time_ns: int | None,
    end_time_ns: int | None,
) -> bool:
    if start_time_ns is not None and event.time_ns < start_time_ns:
        return False
    if end_time_ns is not None and event.time_ns > end_time_ns:
        return False
    return not channels or _channel_matches(event.channel, channels)


def _channel_matches(channel: str, patterns: tuple[str, ...]) -> bool:
    return any(_wildcard_matches(channel, pattern) for pattern in patterns)


def _wildcard_matches(value: str, pattern: str) -> bool:
    value_index = 0
    pattern_index = 0
    star_index = -1
    star_value_index = -1
    while value_index < len(value):
        if pattern_index < len(pattern) and pattern[pattern_index] in {"?", value[value_index]}:
            value_index += 1
            pattern_index += 1
            continue
        if pattern_index < len(pattern) and pattern[pattern_index] == "*":
            star_index = pattern_index
            star_value_index = value_index
            pattern_index += 1
            continue
        if star_index < 0:
            return False
        pattern_index = star_index + 1
        star_value_index += 1
        value_index = star_value_index
    while pattern_index < len(pattern) and pattern[pattern_index] == "*":
        pattern_index += 1
    return pattern_index == len(pattern)


def _validate_observation_response(
    observation: ReplayObservation,
    query: ReplayQuery,
    *,
    image_data: bool,
) -> None:
    if len(observation.events) > query.max_events:
        raise ReplayIntegrityError("replay observation exceeds the requested max_events")
    if len(observation.images) > query.max_images:
        raise ReplayIntegrityError("replay observation exceeds the requested max_images")
    for event in observation.events:
        if query.channels and not _channel_matches(event.channel, query.channels):
            raise ReplayIntegrityError(
                "replay observation contains an event outside the requested channels"
            )
    for image in observation.images:
        if query.channels and not _channel_matches(image.channel, query.channels):
            raise ReplayIntegrityError(
                "replay observation contains an image outside the requested channels"
            )
        if not image_data and image.data is not None:
            raise ReplayIntegrityError(
                "replay observation returned inline image bytes when image_data was omitted"
            )


def _strict_json_loads(text: str, where: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise ReplaySchemaError(f"{where} returned invalid strict JSON") from exc
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if depth > _MAX_JSON_DEPTH:
            raise ReplaySchemaError(f"{where} JSON exceeds the replay nesting limit")
        if isinstance(item, Mapping):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ReplaySchemaError(f"{where} JSON contains a non-finite number")
    return value


def _validate_observation_query(query: ReplayQuery) -> None:
    if query.recording_ids:
        raise ReplaySchemaError(
            "recording_ids is available only for raw iter_events() reads; "
            "the observation endpoint projects all matching recordings"
        )
    if not query.include_control_events:
        raise ReplaySchemaError(
            "include_control_events=False is available only for raw iter_events() reads; "
            "filter the observation endpoint by channel instead"
        )


def _resource_id(value: str, where: str) -> str:
    if not isinstance(value, str) or _RESOURCE_ID_RE.fullmatch(value) is None:
        raise ReplaySchemaError(
            f"{where} must be a safe resource ID matching {_RESOURCE_ID_RE.pattern!r} "
            "with at most 256 characters"
        )
    return value


def _path_segment(value: str, where: str) -> str:
    return quote(_resource_id(value, where), safe="")


def _optional_page_string(value: Any, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ReplaySchemaError(f"{where} must be a non-empty string or null")
    return value


def _query_time_ns(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplaySchemaError(f"{where} must be a Python integer")
    if value < 0 or value > _MAX_TIME_NS:
        raise ReplaySchemaError(f"{where} must be between 0 and {_MAX_TIME_NS}")
    return value
