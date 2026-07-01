"""Per-measurement label verification -- the trust score the deliverable renders on.

Closes the "map labels don't match the manual DWG" gap WITHIN OCR's honest limits:
score each dimension label by (a) agreement with an external exact measurement when
supplied, else (b) self-consistency with its anchored edge. Geometry is never
touched; only ``Measurement.label_confidence`` is set. Never a gate.
"""
from __future__ import annotations

from landintel.agent.label_verify import (GROUND_TRUTH_TOL_M, verify_labels,
                                          inconsistent_measurements)
from landintel.core.models import Boundary, CornerPoint, Measurement, Plot


def _plot(measurements: list[Measurement]) -> Plot:
    return Plot(
        client_id="c", survey_no="42", district="D", taluk="T", village="V",
        scale=2000, stated_area=1.0,
        boundary=Boundary(points=[(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]),
        corner_points=[CornerPoint(label=str(i), x=float(i), y=0.0) for i in range(4)],
        measurements=measurements,
    )


def test_consistent_value_scores_high_and_sets_field():
    m = Measurement(raw="40.0", value=40.0, line_length_m=40.2)
    plot = _plot([m])
    report = verify_labels(plot)
    assert report.consistent == 1 and report.inconsistent == 0
    assert m.label_confidence is not None and m.label_confidence > 0.9


def test_inconsistent_value_scores_low_but_is_not_a_gate():
    m = Measurement(raw="98", value=98.0, line_length_m=40.0)
    plot = _plot([m])
    report = verify_labels(plot)
    assert report.inconsistent == 1
    assert m.label_confidence is not None and m.label_confidence < 0.5
    assert "98" in report.inconsistent_raws


def test_unparseable_token_is_lowest_trust():
    m = Measurement(raw="Y8t6", value=None, line_length_m=40.0)
    plot = _plot([m])
    verify_labels(plot)
    assert m.label_confidence == 0.0


def test_ground_truth_confirms_and_offers_correction():
    # OCR read 44.0 on a 44.18 m surveyed edge -> confirmed + corrected to exact.
    m = Measurement(raw="44,0", value=44.0, line_length_m=39.0)  # edge disagrees...
    plot = _plot([m])
    report = verify_labels(plot, ground_truth_edges=[12.3, 44.18, 90.0])
    # ...but the exact ground truth (44.18, within tol) overrides: confirmed.
    assert report.confirmed == 1 and report.corrections == 1
    assert m.label_confidence == 1.0
    v = report.verifications[0]
    assert v.status == "confirmed" and abs(v.corrected_value - 44.18) < 1e-9


def test_ground_truth_too_far_does_not_confirm():
    m = Measurement(raw="44", value=44.0, line_length_m=44.1)
    plot = _plot([m])
    # Nearest exact is 50.0, gap 6 m >> GROUND_TRUTH_TOL_M -> falls back to geometry.
    report = verify_labels(plot, ground_truth_edges=[50.0, 90.0])
    assert report.confirmed == 0
    assert report.consistent == 1  # 44 vs 44.1 edge is consistent
    assert GROUND_TRUTH_TOL_M < 1.0


def test_trusted_fraction_and_shared_diagnostic():
    plot = _plot([
        Measurement(raw="40.0", value=40.0, line_length_m=40.1),  # consistent
        Measurement(raw="98", value=98.0, line_length_m=40.0),    # inconsistent
        Measurement(raw="x", value=None),                          # unparseable
    ])
    report = verify_labels(plot)
    # 1 of 2 valued measurements is trusted.
    assert abs(report.trusted_fraction - 0.5) < 1e-9
    # The standalone helper agrees with the report (single source of truth).
    assert inconsistent_measurements(_plot([
        Measurement(raw="98", value=98.0, line_length_m=40.0)])) == ["98"]
