from __future__ import annotations

from cybernetics.lib.public_interfaces.service_client import ServiceClient


class _FakeFuture:
    def __init__(self, value: object) -> None:
        self.value = value
        self.timeout = None

    def result(self, timeout=None):  # type: ignore[no-untyped-def]
        self.timeout = timeout
        return self.value


class _FakeHolder:
    def get_telemetry(self):  # type: ignore[no-untyped-def]
        return None


def test_create_lora_training_client_forwards_timeout() -> None:
    client = object.__new__(ServiceClient)
    client.holder = _FakeHolder()  # type: ignore[assignment]
    future = _FakeFuture("training")

    def _submit(*args):  # type: ignore[no-untyped-def]
        return future

    client._create_lora_training_client_submit = _submit  # type: ignore[method-assign]

    result = client.create_lora_training_client("dreamzero-droid", timeout=12.5)

    assert result == "training"
    assert future.timeout == 12.5


def test_create_sampling_client_forwards_timeout(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from cybernetics.lib.public_interfaces.sampling_client import SamplingClient

    client = object.__new__(ServiceClient)
    client.holder = _FakeHolder()  # type: ignore[assignment]
    future = _FakeFuture("sampling")

    def _create(holder, *, model_path, base_model, retry_config):  # type: ignore[no-untyped-def]
        assert holder is client.holder
        assert model_path is None
        assert base_model == "dreamzero-droid"
        assert retry_config is None
        return future

    monkeypatch.setattr(SamplingClient, "create", _create)

    result = client.create_sampling_client(base_model="dreamzero-droid", timeout=7)

    assert result == "sampling"
    assert future.timeout == 7
