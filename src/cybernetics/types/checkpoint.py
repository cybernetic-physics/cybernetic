from datetime import datetime
from typing import Literal

from .._models import BaseModel

__all__ = ["Checkpoint", "CheckpointType"]

CheckpointType = Literal["training", "sampler"]


class Checkpoint(BaseModel):
    checkpoint_id: str
    """The checkpoint ID"""

    checkpoint_type: CheckpointType
    """The type of checkpoint (training or sampler)"""

    time: datetime
    """The time when the checkpoint was created"""

    worldlines_path: str
    """The worldlines path to the checkpoint"""

    size_bytes: int | None = None
    """The size of the checkpoint in bytes"""

    public: bool = False
    """Whether the checkpoint is publicly accessible"""

    expires_at: datetime | None = None
    """When this checkpoint expires (None = never expires)"""


class ParsedCheckpointCyberneticsPath(BaseModel):
    worldlines_path: str
    """The worldlines path to the checkpoint"""

    training_run_id: str
    """The training run ID"""

    checkpoint_type: CheckpointType
    """The type of checkpoint (training or sampler)"""

    checkpoint_id: str
    """The checkpoint ID"""

    @classmethod
    def from_worldlines_path(cls, worldlines_path: str) -> "ParsedCheckpointCyberneticsPath":
        """Parse a worldlines path to an instance of ParsedCheckpointCyberneticsPath"""
        if not worldlines_path.startswith("worldlines://"):
            raise ValueError(f"Invalid worldlines path: {worldlines_path}")
        parts = worldlines_path.removeprefix("worldlines://").split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid worldlines path: {worldlines_path}")
        if parts[1] not in ["weights", "sampler_weights"]:
            raise ValueError(f"Invalid worldlines path: {worldlines_path}")
        checkpoint_type = "training" if parts[1] == "weights" else "sampler"
        return cls(
            worldlines_path=worldlines_path,
            training_run_id=parts[0],
            checkpoint_type=checkpoint_type,
            checkpoint_id="/".join(parts[1:]),
        )
