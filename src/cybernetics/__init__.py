import typing as _t

from . import types
from ._base_client import DefaultAioHttpClient, DefaultAsyncHttpxClient
from ._client import AsyncCybernetics, RequestOptions, Timeout
from ._exceptions import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    CyberneticsError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RequestFailedError,
    SidecarDiedError,
    SidecarError,
    SidecarIPCError,
    SidecarStartupError,
    UnprocessableEntityError,
)
from ._response import APIResponse as APIResponse
from ._response import AsyncAPIResponse as AsyncAPIResponse
from ._utils._logs import setup_logging as _setup_logging
from ._version import __title__, __version__
from .client import Client
from .lib.public_interfaces import APIFuture, SamplingClient, ServiceClient, TrainingClient
from .replay import ReplayClient, ReplayError, ReplayEventSelection, ReplayObservation, ReplayQuery

# The generated SDK is async-first; keep the conventional top-level client name
# available for callers that import it from the package root.
Cybernetics = AsyncCybernetics

# Import commonly used types for easier access
from .types import (
    AdamParams,
    Checkpoint,
    CheckpointType,
    Datum,
    DroidObservation,
    EncodedTextChunk,
    ForwardBackwardOutput,
    LoraConfig,
    ModelID,
    ModelInput,
    ModelInputChunk,
    OptimStepRequest,
    OptimStepResponse,
    ParsedCheckpointCyberneticsPath,
    Pi0DroidDsrlAction,
    SampledSequence,
    SampleRequest,
    SampleResponse,
    SamplingParams,
    StopReason,
    TensorData,
    TensorDtype,
    TrainingRun,
)

__all__ = [
    # Core clients
    "TrainingClient",
    "ServiceClient",
    "SamplingClient",
    "APIFuture",
    "AsyncCybernetics",
    "Cybernetics",
    "Client",
    "ReplayClient",
    "ReplayError",
    "ReplayEventSelection",
    "ReplayObservation",
    "ReplayQuery",
    # Commonly used types
    "AdamParams",
    "Checkpoint",
    "CheckpointType",
    "Datum",
    "DroidObservation",
    "EncodedTextChunk",
    "ForwardBackwardOutput",
    "LoraConfig",
    "ModelID",
    "ModelInput",
    "ModelInputChunk",
    "OptimStepRequest",
    "OptimStepResponse",
    "ParsedCheckpointCyberneticsPath",
    "Pi0DroidDsrlAction",
    "SampledSequence",
    "SampleRequest",
    "SampleResponse",
    "SamplingParams",
    "StopReason",
    "TensorData",
    "TensorDtype",
    "TrainingRun",
    # Client configuration
    "Timeout",
    "RequestOptions",
    "DefaultAioHttpClient",
    "DefaultAsyncHttpxClient",
    # Exception types
    "CyberneticsError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "APIConnectionError",
    "APIResponseValidationError",
    "RequestFailedError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "SidecarError",
    "SidecarStartupError",
    "SidecarDiedError",
    "SidecarIPCError",
    # Keep types module for advanced use
    "types",
    # Version info
    "__version__",
    "__title__",
]

if not _t.TYPE_CHECKING:
    from ._utils._resources_proxy import resources as resources

_setup_logging()

# Update the __module__ attribute for exported symbols so that
# error messages point to this module instead of the module
# it was originally defined in, e.g.
# cybernetics._exceptions.NotFoundError -> cybernetics.NotFoundError
__locals = locals()
for __name in __all__:
    if not __name.startswith("__"):
        try:
            __locals[__name].__module__ = "cybernetics"
        except (TypeError, AttributeError):
            # Some of our exported symbols are builtins which we can't set attributes for.
            pass
