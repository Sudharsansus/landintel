"""Orchestrator tests: M1+agent wired end to end on a real fixture.

Tests that the built pipeline (M1 + validate + anomaly + audit + DXF write)
runs start to finish producing a real Job with plots and output files. Also
verifies the M2/M3/M4 stubs raise explicitly rather than silently passing, and
that a GeometryError on one PDF is caught and logged without crashing the job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from landintel.core.enums import JobStatus, PlotStatus, Stage
from landintel.core.exceptions import GeometryError
from landintel.core.models import Job
from landintel.pipeline.orchestrator import (
    _m2_club,
    _m2_georef,
    _m3_assemble,
    _m4_report,
    run_job,
)

FMB_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "FMB"


def make_job(pdfs: list[str]) -> Job:
    return Job(
        client_id="client_test",
        input_files=pdfs,
    )


# --- M2 is built; M3/M4 stubs still raise explicitly -------------------------


def test_m2_is_built_not_a_stub(tmp_path: Path) -> None:
    """M2 is wired (no longer a NotImplementedError stub). With no M1 DXFs it
    returns an empty batch without touching any surveyor file."""
    results = _m2_georef([], None, tmp_path)  # type: ignore[arg-type]
    assert results == []


def test_m2_club_empty_batch(tmp_path: Path) -> None:
    """The NEW M2 (FMB-only club) returns an empty batch for no inputs."""
    assert _m2_club([], tmp_path) == []


class _NullCadastral:
    """A cadastral source that has no parcel for any survey (forces NO_COVERAGE)."""

    def get(self, survey, village=None):
        return None

    def label_point(self, survey):
        return None

    def recovered_candidates(self, survey):
        return []


def test_m2_club_stage_runs_opt_in_without_crashing(tmp_path: Path) -> None:
    """When a cadastral source is supplied, run_job runs the M2-club stage after M1.
    With a null source every plot is honest NO_COVERAGE (0-FP), the job does not
    crash, the agent layer runs, and the job still reaches DELIVERED."""
    pdf = str(FMB_DIR / "FMB_SIVAGANGAI_Manamadurai_T.Pudukkottai_31.pdf")
    job = make_job([pdf])
    result = run_job(job, output_dir=tmp_path / "out", agent_client=None,
                     cadastral_source=_NullCadastral())
    assert result.stage is Stage.DELIVERED
    assert any("M2 club" in line for line in result.audit)


def test_m3_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="M3 assemble"):
        _m3_assemble([], Path("."))


def test_m4_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="M4 report"):
        _m4_report(None, Path("."))  # type: ignore[arg-type]


# --- Real end-to-end: M1 + agent on one fixture ------------------------------


@pytest.fixture(scope="module")
def job_result(tmp_path_factory) -> Job:
    """Run survey 31 (small/fast) through the full built chain."""
    pdf = str(FMB_DIR / "FMB_SIVAGANGAI_Manamadurai_T.Pudukkottai_31.pdf")
    job = make_job([pdf])
    out = tmp_path_factory.mktemp("orchestrator_out")
    return run_job(job, output_dir=out, agent_client=None)


def test_job_has_one_plot(job_result: Job) -> None:
    assert len(job_result.plots) == 1
    assert job_result.plots[0].survey_no == "31"
    assert job_result.plots[0].client_id == "client_test"


def test_job_stage_is_delivered(job_result: Job) -> None:
    """Stage is DELIVERED — M1+agent ran; M2/M3/M4 are stubs (not invoked yet)."""
    assert job_result.stage is Stage.DELIVERED


def test_job_status_derives_correctly(job_result: Job) -> None:
    """Status is derived from plot statuses — COMPLETED or NEEDS_REVIEW after M1."""
    assert job_result.status in (JobStatus.COMPLETED, JobStatus.NEEDS_REVIEW)


def test_dxf_output_was_written(job_result: Job) -> None:
    assert len(job_result.output_files) == 1
    dxf_path = Path(job_result.output_files[0])
    assert dxf_path.exists() and dxf_path.suffix == ".dxf"
    assert dxf_path.stat().st_size > 0


def test_audit_trail_has_entry(job_result: Job) -> None:
    assert len(job_result.audit) >= 1
    assert "31" in job_result.audit[0]


def test_plot_boundary_is_closed_with_real_area(job_result: Job) -> None:
    plot = job_result.plots[0]
    assert plot.boundary is not None and plot.boundary.is_closed
    computed_ha = plot.boundary.computed_area / 10_000.0
    # Survey 31 stated 1.115 ha; we consistently land within 10%.
    assert plot.stated_area is not None
    assert abs(computed_ha - plot.stated_area) / plot.stated_area < 0.10


# --- GeometryError on one PDF is caught, job continues ----------------------


def test_geometry_error_on_one_pdf_does_not_crash_job(tmp_path: Path) -> None:
    """A PDF that fails geometry extraction marks that plot FAILED; job still runs."""
    real_pdf = str(FMB_DIR / "FMB_SIVAGANGAI_Manamadurai_T.Pudukkottai_31.pdf")
    fake_pdf = str(tmp_path / "nonexistent.pdf")  # will raise on open

    job = make_job([fake_pdf, real_pdf])
    result = run_job(job, output_dir=tmp_path / "out", agent_client=None)

    # The real PDF still produced a plot.
    assert any(p.survey_no == "31" for p in result.plots)
    # The broken "plot" was recorded in the audit trail.
    assert any("ERROR" in line or "FAILED" in line for line in result.audit)
    # Job did not raise — it completed the run.
    assert result.stage is Stage.DELIVERED
