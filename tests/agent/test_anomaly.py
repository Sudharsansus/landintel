"""Tests for the anomaly layer: geometry gates on real fixtures + edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from landintel.agent.anomaly import check_plot
from landintel.agent.validator import validate_plot
from landintel.core.enums import PlotStatus
from landintel.core.models import Boundary, CornerPoint, Measurement, Plot
from landintel.pipeline.m1_extract.anchor import anchor_measurements
from landintel.pipeline.m1_extract.build_plot import build_plot
from landintel.pipeline.m1_extract.ocr import extract_text, parse_header
from landintel.pipeline.m1_extract.pdf_vectors import extract_vectors

FMB_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "FMB"


def square(side: float) -> Boundary:
    return Boundary(points=[(0.0, 0.0), (side, 0.0), (side, side), (0.0, side), (0.0, 0.0)])


def make_plot(boundary: Boundary, stated_area: float, n_stones: int = 4) -> Plot:
    return Plot(
        client_id="c", survey_no="42", district="D", taluk="T", village="V",
        scale=2000, stated_area=stated_area, boundary=boundary,
        corner_points=[CornerPoint(label=str(i), x=float(i), y=0.0) for i in range(n_stones)],
    )


# --- Constructed edge cases (fast, exact) ------------------------------------


def test_clean_plot_passes() -> None:
    plot = make_plot(square(100.0), stated_area=1.0)  # 100x100 m = 1.0 ha
    report = check_plot(plot)
    assert report.ok and not report.failed
    assert plot.status is not PlotStatus.FAILED


def test_open_boundary_flags_not_crashes() -> None:
    open_b = Boundary(points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)])  # no closing edge
    plot = make_plot(open_b, stated_area=1.0)
    report = check_plot(plot)
    assert not report.ok and not report.failed       # flag, not fail
    assert any(i.code == "non_closing" for i in report.issues)
    assert plot.status is PlotStatus.FLAGGED


def test_area_mismatch_within_band_flags() -> None:
    plot = make_plot(square(100.0), stated_area=1.2)  # 1.0 vs 1.2 ha = 16.7%
    report = check_plot(plot)
    assert any(i.code == "area_mismatch" and i.severity == "flag" for i in report.issues)
    assert plot.status is PlotStatus.FLAGGED


def test_gross_area_mismatch_fails() -> None:
    plot = make_plot(square(100.0), stated_area=3.0)  # 1.0 vs 3.0 ha = 66%
    report = check_plot(plot)
    assert report.failed
    assert any(i.code == "area_mismatch" and i.severity == "fail" for i in report.issues)
    assert plot.status is PlotStatus.FAILED


def test_too_few_stones_flags() -> None:
    plot = make_plot(square(100.0), stated_area=1.0, n_stones=2)
    report = check_plot(plot)
    assert any(i.code == "too_few_stones" for i in report.issues)


def test_inconsistent_measurement_is_reported_not_gated() -> None:
    """A mis-anchored value is surfaced as a diagnostic, but does NOT fail/flag."""
    plot = make_plot(square(100.0), stated_area=1.0)
    # "98" on a 40 m edge: value disagrees with edge length -> diagnostic only.
    plot.measurements = [Measurement(raw="98", value=98.0, line_length_m=40.0)]
    report = check_plot(plot)
    assert "98" in report.inconsistent_measurements
    assert report.ok  # geometry is clean; the inconsistency is not a gate


# --- Real fixtures (geometry must pass clean) --------------------------------


def build_and_validate(survey: int) -> Plot:
    f = FMB_DIR / f"FMB_SIVAGANGAI_Manamadurai_T.Pudukkottai_{survey}.pdf"
    vectors = extract_vectors(f)
    detections = extract_text(f)
    plot = build_plot(
        client_id="c", vectors=vectors, detections=detections,
        anchor_result=anchor_measurements(vectors, detections),
        header=parse_header(detections),
    )
    validate_plot(plot, client=None)
    return plot


@pytest.fixture(scope="module")
def real_plots() -> dict[int, Plot]:
    return {s: build_and_validate(s) for s in (100, 199, 31)}


@pytest.mark.parametrize("survey", [100, 199, 31])
def test_real_fixtures_pass_geometry_checks(real_plots: dict[int, Plot], survey: int) -> None:
    """All three clean fixtures close, match area, and have enough stones."""
    report = check_plot(real_plots[survey])
    assert not report.failed
    codes = {i.code for i in report.issues}
    assert codes.isdisjoint({"no_boundary", "non_closing", "area_mismatch", "too_few_stones"})
    assert report.ok


def test_real_fixture_reports_measurement_noise_diagnostic(real_plots: dict[int, Plot]) -> None:
    """The known measurement-label noise is surfaced as a diagnostic, not a gate."""
    report = check_plot(real_plots[100])
    assert len(report.inconsistent_measurements) > 0
    assert report.ok  # but geometry still passes
