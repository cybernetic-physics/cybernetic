from pydantic import Field

from .._models import BaseModel

__all__ = ["WorldlinesSessionCancelResponse"]


class WorldlinesSessionCancelResponse(BaseModel):
    ok: bool
    stopped_lease_ids: list[str] = Field(default_factory=list)
