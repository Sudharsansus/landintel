"""Human review routes — submit corrections on flagged plots."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...agent.memory import build_correction
from ...db.repositories.corrections import CorrectionRepository
from ...db.repositories.plots import PlotRepository
from ...core.enums import PlotStatus
from ..deps import current_client, get_correction_repo, get_plot_repo
from ..schemas.plot import PlotDetail, PlotSummary
from ..schemas.review import CorrectionCreate, CorrectionResponse

router = APIRouter(prefix="/review", tags=["review"])


@router.get("/flagged", response_model=list[PlotSummary])
async def list_flagged_plots(
    client_id: str = Depends(current_client),
    plot_repo: PlotRepository = Depends(get_plot_repo),
) -> list[PlotSummary]:
    """List all plots currently awaiting human review."""
    plots = await plot_repo.list_by_status(client_id, PlotStatus.FLAGGED)
    return [PlotSummary(
        survey_no=p.survey_no,
        status=p.status,
        stated_area=p.stated_area,
        flags=p.flags,
    ) for p in plots]


@router.get("/{survey_no}", response_model=PlotDetail)
async def get_plot_for_review(
    survey_no: str,
    client_id: str = Depends(current_client),
    plot_repo: PlotRepository = Depends(get_plot_repo),
) -> PlotDetail:
    plot = await plot_repo.get(client_id, survey_no)
    if plot is None:
        raise HTTPException(status_code=404, detail="plot not found")
    return PlotDetail.from_plot(plot)


@router.post("/{survey_no}/corrections", response_model=CorrectionResponse)
async def submit_correction(
    survey_no: str,
    body: CorrectionCreate,
    job_id: str,
    client_id: str = Depends(current_client),
    plot_repo: PlotRepository = Depends(get_plot_repo),
    correction_repo: CorrectionRepository = Depends(get_correction_repo),
) -> CorrectionResponse:
    """Record a human correction on a measurement or field."""
    plot = await plot_repo.get(client_id, survey_no)
    if plot is None:
        raise HTTPException(status_code=404, detail="plot not found")

    correction = build_correction(
        client_id=client_id,
        job_id=job_id,
        plot_id=survey_no,
        field=body.field,
        old=body.old,
        new=body.new,
        measurement_ref=body.measurement_ref,
    )
    await correction_repo.record(client_id, correction)

    return CorrectionResponse(
        id=correction.id,
        plot_id=survey_no,
        field=body.field,
        old=body.old,
        new=body.new,
    )
