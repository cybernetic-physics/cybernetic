from typing import Optional

from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["SaveWeightsResponse"]


class SaveWeightsResponse(BaseModel):
    path: str
    """A worldlines URI for model weights at a specific step"""

    has_optimizer_state: bool | None = None
    """Whether the checkpoint includes optimizer state for training resume"""

    type: Optional[Literal["save_weights"]] = None
