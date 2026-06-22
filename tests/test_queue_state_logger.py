from __future__ import annotations

import logging

from cybernetics.lib.api_future_impl import QueueState
from cybernetics.lib.queue_state_logger import QueueStateLogger


def test_active_progress_reason_is_logged(caplog) -> None:  # type: ignore[no-untyped-def]
    logger = QueueStateLogger("dreamzero/droid", "Model creation")

    with caplog.at_level(logging.WARNING, logger="cybernetics.lib.queue_state_logger"):
        logger.on_queue_state_change(QueueState.ACTIVE, "dreamzero:build_vla")

    assert "Model creation for dreamzero/droid is running" in caplog.text
    assert "Progress: dreamzero:build_vla" in caplog.text


def test_active_without_reason_stays_quiet(caplog) -> None:  # type: ignore[no-untyped-def]
    logger = QueueStateLogger("dreamzero/droid", "Model creation")

    with caplog.at_level(logging.WARNING, logger="cybernetics.lib.queue_state_logger"):
        logger.on_queue_state_change(QueueState.ACTIVE, None)

    assert caplog.text == ""


def test_repeated_identical_progress_reason_is_throttled(caplog) -> None:  # type: ignore[no-untyped-def]
    logger = QueueStateLogger("dreamzero/droid", "Model creation")

    with caplog.at_level(logging.WARNING, logger="cybernetics.lib.queue_state_logger"):
        logger.on_queue_state_change(QueueState.ACTIVE, "dreamzero:build_vla")
        logger.on_queue_state_change(QueueState.ACTIVE, "dreamzero:build_vla")
        logger.on_queue_state_change(QueueState.ACTIVE, "dreamzero:build_optimizer")

    assert caplog.text.count("Progress: dreamzero:build_vla") == 1
    assert caplog.text.count("Progress: dreamzero:build_optimizer") == 1
