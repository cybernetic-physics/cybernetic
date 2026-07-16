"""Composable public Cybernetics SDK client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .replay import ReplayClient


class Client:
    """Small composition root for product-specific SDK namespaces.

    ``Client`` intentionally does not belong to any one namespace. It exposes
    product surfaces such as ``replay`` and ``sim`` without making those modules
    depend on each other.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: Any = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._http_client = http_client
        self._replay: ReplayClient | None = None
        self._sim: Any | None = None

    @property
    def replay(self) -> ReplayClient:
        if self._replay is None:
            from .replay import ReplayClient

            self._replay = ReplayClient(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=self._http_client,
            )
        return self._replay

    @property
    def sim(self) -> Any:
        if self._sim is None:
            from .sim import SimulationClient

            self._sim = SimulationClient(
                api_key=self._api_key,
                base_url=self._base_url,
                http_client=self._http_client,
            )
        return self._sim

    def close(self) -> None:
        first_error: Exception | None = None
        for name, namespace in (("replay", self._replay), ("sim", self._sim)):
            if namespace is None:
                continue
            try:
                namespace.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                else:
                    first_error.add_note(
                        f"Cybernetics {name} cleanup also failed: {type(exc).__name__}: {exc}"
                    )
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self.close()
        except Exception as close_error:
            if isinstance(exc, BaseException):
                exc.add_note(
                    "Cybernetics client cleanup failed while handling the primary error: "
                    f"{type(close_error).__name__}: {close_error}"
                )
                return
            raise
