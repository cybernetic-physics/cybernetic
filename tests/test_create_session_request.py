from __future__ import annotations

from cybernetics._compat import model_dump
from cybernetics.lib.internal_client_holder import _create_session_request


def test_create_session_request_omits_default_project_id() -> None:
    request = _create_session_request(
        tags=[],
        user_metadata=None,
        sdk_version="test-sdk",
        project_id=None,
    )

    payload = model_dump(request, exclude_unset=True, mode="json")

    assert payload == {
        "tags": [],
        "user_metadata": {},
        "sdk_version": "test-sdk",
    }


def test_create_session_request_serializes_explicit_project_id() -> None:
    request = _create_session_request(
        tags=["demo"],
        user_metadata={"source": "test"},
        sdk_version="test-sdk",
        project_id="robotics-demo",
    )

    payload = model_dump(request, exclude_unset=True, mode="json")

    assert payload == {
        "tags": ["demo"],
        "user_metadata": {"source": "test"},
        "sdk_version": "test-sdk",
        "project_id": "robotics-demo",
    }
