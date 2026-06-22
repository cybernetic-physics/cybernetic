"""Contract: the public import surface, narrow exports, and lazy heavy deps.

These assert the headline product guarantees: ``import cybernetics`` and the five
root exports work with neither ``torch`` nor ``transformers`` installed, and
``RestClient`` stays private (reachable only via ``create_rest_client()``).
"""

from __future__ import annotations

import importlib.util


def test_torch_and_transformers_absent_from_test_env() -> None:
    # The whole point of the dependency diet: the core import path must not need them.
    assert importlib.util.find_spec("torch") is None
    assert importlib.util.find_spec("transformers") is None


def test_import_cybernetics_and_root_exports() -> None:
    import cybernetics
    from cybernetics import (  # noqa: F401
        APIFuture,
        SamplingClient,
        ServiceClient,
        TrainingClient,
        types,
    )

    assert cybernetics.__title__ == "cybernetic-physics"
    assert isinstance(cybernetics.__version__, str) and cybernetics.__version__


def test_rest_client_is_not_a_root_export() -> None:
    import cybernetics

    assert not hasattr(cybernetics, "RestClient")
    assert hasattr(cybernetics.ServiceClient, "create_rest_client")


def test_brand_identifiers_renamed() -> None:
    from cybernetics import (  # noqa: F401
        AsyncCybernetics,
        Cybernetics,
        CyberneticsError,
        ParsedCheckpointCyberneticsPath,
    )

    # __module__ rewrite points exported names at the package root.
    assert CyberneticsError.__module__ == "cybernetics"


def test_loss_fn_vocabulary_includes_flow_rwr() -> None:
    """One closed Literal carries the five Tinker losses plus DreamZero's flow_rwr."""

    from typing import get_args

    from cybernetics.types.loss_fn_type import LossFnType

    losses = set(get_args(LossFnType))
    assert {"cross_entropy", "importance_sampling", "ppo", "cispo", "dro"} <= losses
    assert "flow_rwr" in losses  # the one Worldlines/DreamZero divergence from Tinker
