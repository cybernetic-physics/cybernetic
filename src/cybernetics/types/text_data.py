from typing import List, Optional

from typing_extensions import Literal

from .._models import StrictBase

__all__ = ["TextData"]


class TextData(StrictBase):
    """A shaped UTF-8 observation carried beside numeric policy tensors."""

    data: List[str]
    dtype: Literal["utf8"]
    shape: Optional[List[int]] = None
