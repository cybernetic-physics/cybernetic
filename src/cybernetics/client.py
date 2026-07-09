"""Composable public Cybernetics SDK client."""

from __future__ import annotations

from typing import Any


class Client:
    """Small composition root for product-specific SDK namespaces.

    ``Client`` intentionally does not belong to any one namespace. It can expose
    ``sim`` today and other product surfaces later without making those modules
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
        self._sim: Any | None = None

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
        if self._sim is not None:
            self._sim.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
