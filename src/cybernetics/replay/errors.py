"""Replay SDK exceptions."""


class ReplayError(RuntimeError):
    """A durable session replay operation failed."""


class ReplaySchemaError(ReplayError, ValueError):
    """A replay API or artifact payload violates its declared schema."""


class ReplayIntegrityError(ReplayError):
    """Downloaded replay bytes do not match their immutable chunk index."""
