"""Durable Cybernetics session replay and agent-context helpers.

The canonical replay types are provider-neutral. OpenAI and Anthropic adapters
return plain dictionaries and do not import either provider SDK.
"""

from .client import ReplayClient
from .errors import ReplayError, ReplayIntegrityError, ReplaySchemaError
from .models import (
    AGENT_REPLAY_BUNDLE_SCHEMA_VERSION,
    AGENT_REPLAY_OBSERVATION_SCHEMA_VERSION,
    REPLAY_EVENT_SELECTION_SCHEMA_VERSION,
    REPLAY_OBSERVATION_SCHEMA_VERSION,
    AgentReplayBundle,
    AgentReplayOptions,
    ReplayChunk,
    ReplayEvent,
    ReplayEventSelection,
    ReplayImage,
    ReplayObservation,
    ReplayQuery,
    ReplayRecording,
    ReplaySummary,
)

__all__ = [
    "AGENT_REPLAY_OBSERVATION_SCHEMA_VERSION",
    "AGENT_REPLAY_BUNDLE_SCHEMA_VERSION",
    "REPLAY_OBSERVATION_SCHEMA_VERSION",
    "REPLAY_EVENT_SELECTION_SCHEMA_VERSION",
    "AgentReplayBundle",
    "AgentReplayOptions",
    "ReplayChunk",
    "ReplayClient",
    "ReplayError",
    "ReplayEvent",
    "ReplayEventSelection",
    "ReplayImage",
    "ReplayIntegrityError",
    "ReplayObservation",
    "ReplayQuery",
    "ReplayRecording",
    "ReplaySchemaError",
    "ReplaySummary",
]
