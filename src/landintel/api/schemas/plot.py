"""API schemas for plots."""

from __future__ import annotations

from pydantic import BaseModel

from ...core.enums import PlotStatus

__all__ = ["PlotSummary", "PlotDetail"]


class PlotSummary(BaseModel):
    """Minimal plot info for list views."""
    survey_no: str
    status: PlotStatus
    stated_area: float | None
    flags: list[str]


class PlotDetail(PlotSummary):
    """Full plot detail including measurements and corner counts."""
    district: str
    taluk: str
    village: str
    scale: int | None
    measurement_count: int
    corner_count: int
    boundary_closed: bool | None

    @classmethod
    def from_plot(cls, plot) -> "PlotDetail":
        return cls(
            survey_no=plot.survey_no,
            status=plot.status,
            stated_area=plot.stated_area,
            flags=plot.flags,
            district=plot.district,
            taluk=plot.taluk,
            village=plot.village,
            scale=plot.scale,
            measurement_count=len(plot.measurements),
            corner_count=len(plot.corner_points),
            boundary_closed=plot.boundary.is_closed if plot.boundary else None,
        )
