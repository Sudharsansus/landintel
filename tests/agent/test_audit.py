"""Tests for the natural-language audit trail."""

from __future__ import annotations

from landintel.agent.anomaly import AnomalyReport, check_plot
from landintel.agent.audit import audit_plot
from landintel.core.enums import PlotStatus
from landintel.core.models import Boundary, CornerPoint, Measurement, Plot


def square_plot(stated_area: float) -> Plot:
    return Plot(
        client_id="c", survey_no="100", district="D", taluk="T", village="V",
        scale=2000, stated_area=stated_area,
        boundary=Boundary(points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)]),
        corner_points=[CornerPoint(label=str(i), x=float(i), y=0.0) for i in range(4)],
        measurements=[
            Measurement(raw="40,0", value=40.0, confidence=0.95),
            Measurement(raw="Y8t6", confidence=0.5),  # left unaccepted
        ],
    )


def test_audit_line_for_clean_plot() -> None:
    plot = square_plot(stated_area=1.0)
    plot.status = PlotStatus.VALIDATED
    line = audit_plot(plot)
    assert line.startswith("Survey 100:")
    assert "closed" in line
    assert "1.00 ha vs 1.00 stated" in line
    assert "1/2 measurements accepted" in line
    assert "4 stones" in line
    assert line.endswith("— OK")


def test_audit_line_reflects_flags_and_report() -> None:
    plot = square_plot(stated_area=1.0)
    plot.flags = ["[non_closing] ...", "[area_mismatch] ..."]
    plot.status = PlotStatus.FLAGGED
    report = AnomalyReport(inconsistent_measurements=["98", "200"])
    line = audit_plot(plot, report=report)
    assert "FLAGGED (2 issues)" in line
    assert "2 measurement(s) inconsistent with edges" in line


def test_audit_line_for_open_boundary() -> None:
    plot = Plot(
        client_id="c", survey_no="31", district="D", taluk="T", village="V",
        scale=2000, stated_area=1.0,
        boundary=Boundary(points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]),
    )
    check_plot(plot)  # marks it open / flagged
    line = audit_plot(plot)
    assert "OPEN" in line
    assert line.startswith("Survey 31:")
