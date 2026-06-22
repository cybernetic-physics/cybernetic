"""Contract: there is exactly ONE TrainingClient, and SFT vs RL is selected purely
by the ``loss_fn`` argument + the ``Datum.loss_fn_inputs`` keys -- never a second
client type or method. The wire submit is mocked (captured), per the repo's
convention that full HTTP/DB round-trips live in the hosted integration suite.
"""

from __future__ import annotations

import asyncio

from cybernetics import types
from cybernetics.lib.public_interfaces.training_client import TrainingClient


class _Captured(Exception):
    """Raised by the mocked wire submit after recording the request."""


class _FakeHolder:
    """Minimal InternalClientHolder stand-in: runs coroutines in-process and treats
    the wire as mocked. The recorded ``calls`` are the assertion surface."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def server_supports_training(self) -> bool:
        return True

    def estimate_bytes_count_in_model_input(self, _model_input: object) -> int:
        return 8

    async def execute_with_retries(self, fn, *args):  # type: ignore[no-untyped-def]
        return await fn(*args)

    def run_coroutine_threadsafe(self, coro):  # type: ignore[no-untyped-def]
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def get_telemetry(self):  # type: ignore[no-untyped-def]
        return None


def _sft_datum() -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[1, 2, 3], dtype="int64", shape=[3]),
            "weights": types.TensorData(data=[1.0, 1.0, 1.0], dtype="float32", shape=[3]),
        },
    )


def _rl_datum() -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=[1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=[1, 2, 3], dtype="int64", shape=[3]),
            "logprobs": types.TensorData(data=[-0.1, -0.2, -0.3], dtype="float32", shape=[3]),
            "advantages": types.TensorData(data=[1.0, 0.5, -0.5], dtype="float32", shape=[3]),
        },
    )


def _client_with_capture() -> tuple[TrainingClient, _FakeHolder]:
    holder = _FakeHolder()
    client = TrainingClient(holder, model_seq_id=0, model_id="model_x")

    async def _capture(request_id, data, loss_fn, loss_fn_config=None):  # type: ignore[no-untyped-def]
        holder.calls.append((loss_fn, sorted(data[0].loss_fn_inputs.keys())))
        raise _Captured()

    client._send_single_forward_backward_request = _capture  # type: ignore[method-assign]
    return client, holder


def _drive(client: TrainingClient, datum: types.Datum, loss_fn: str) -> None:
    try:
        client.forward_backward([datum], loss_fn)
    except Exception:  # noqa: BLE001 -- the mocked submit raises after recording
        pass


def test_single_training_client_factory_and_type() -> None:
    # No separate SFT/RL client class exists; create_lora_training_client is the one factory.
    assert hasattr(types, "Datum")
    assert TrainingClient.__name__ == "TrainingClient"
    assert hasattr(TrainingClient, "forward_backward")
    # SamplingClient is a distinct role, not a second *training* client.
    from cybernetics import SamplingClient

    assert SamplingClient is not TrainingClient


def test_sft_cross_entropy_routes_through_one_client() -> None:
    client, holder = _client_with_capture()
    _drive(client, _sft_datum(), "cross_entropy")
    assert holder.calls == [("cross_entropy", ["target_tokens", "weights"])]


def test_rl_importance_sampling_and_ppo_route_through_the_same_client() -> None:
    client, holder = _client_with_capture()
    _drive(client, _rl_datum(), "importance_sampling")
    _drive(client, _rl_datum(), "ppo")
    assert holder.calls == [
        ("importance_sampling", ["advantages", "logprobs", "target_tokens"]),
        ("ppo", ["advantages", "logprobs", "target_tokens"]),
    ]


def test_one_client_dispatches_sft_then_rl_by_loss_fn_only() -> None:
    """The SAME instance handles SFT then RL, selected by loss_fn + Datum keys."""

    client, holder = _client_with_capture()
    _drive(client, _sft_datum(), "cross_entropy")
    _drive(client, _rl_datum(), "ppo")
    _drive(client, _rl_datum(), "flow_rwr")  # DreamZero reward-weighted loss accepted
    assert [loss_fn for loss_fn, _ in holder.calls] == ["cross_entropy", "ppo", "flow_rwr"]
