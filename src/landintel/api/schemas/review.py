"""API schemas for human review corrections."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["CorrectionCreate", "CorrectionResponse"]


class CorrectionCreate(BaseModel):
    """A human override for one measurement or field."""
    field: str
    old: str
    new: str
    measurement_ref: str | None = None


class CorrectionResponse(BaseModel):
    id: str
    plot_id: str
    field: str
    old: str
    new: str
