"""Self-calibrating ACCEPT gate -- adapt the chain-coverage bar per span, tighten-only.

The 0-FP invariant under test: calibration can only RAISE the bar (demote ACCEPT ->
REVIEW), never lower it. INGUR-like distributions where the gap sits at the floor are
unchanged; a new-district distribution whose coincidental cluster creeps above the
floor pushes the bar up into the clean gap.
"""
from __future__ import annotations

from landintel.pipeline.m2_georef.self_calibrate import (
    apply_calibrated_gate, calibrate_coverage_threshold)


def test_too_few_samples_returns_floor():
    cal = calibrate_coverage_threshold([0.6, 0.7], floor=0.50)
    assert cal.threshold == 0.50 and not cal.calibrated


def test_ingur_like_gap_at_floor_is_unchanged():
    # Coincidental 12-43%, true 58-100% -> the natural gap straddles 0.50.
    cov = [0.12, 0.20, 0.31, 0.43, 0.58, 0.72, 0.85, 1.0]
    cal = calibrate_coverage_threshold(cov, floor=0.50)
    # The midpoint of the [0.43, 0.58] gap is ~0.505 ~ floor; never drops below floor.
    assert cal.threshold >= 0.50
    assert cal.threshold <= 0.56


def test_strong_coincidental_cluster_raises_the_bar():
    # New district: coincidental matches creep up to 0.55 (ABOVE the static floor),
    # true matches start at 0.74. The clean gap is [0.55, 0.74] -> bar ~0.645.
    cov = [0.20, 0.40, 0.52, 0.55, 0.74, 0.80, 0.91, 0.99]
    cal = calibrate_coverage_threshold(cov, floor=0.50)
    assert cal.calibrated
    assert 0.60 < cal.threshold < 0.70


def test_never_exceeds_ceil():
    cov = [0.50, 0.52, 0.55, 0.58, 0.97, 0.98, 0.99, 1.0]
    cal = calibrate_coverage_threshold(cov, floor=0.50, ceil=0.95)
    assert cal.threshold <= 0.95


def test_apply_gate_demotes_only_below_bar():
    # Mimic GeorefResult rows with a coincidental cluster above floor.
    class R:
        def __init__(self, rec, cov):
            self.recommendation, self.cov = rec, cov

    rows = [
        R("ACCEPT", 0.52), R("ACCEPT", 0.55),   # coincidental, above floor
        R("ACCEPT", 0.78), R("ACCEPT", 0.85),
        R("ACCEPT", 0.93), R("REVIEW", 0.30),   # already REVIEW, untouched
    ]
    n = apply_calibrated_gate(
        rows,
        get_recommendation=lambda r: r.recommendation,
        set_review=lambda r, bar: setattr(r, "recommendation", "REVIEW"),
        get_coverage=lambda r: r.cov,
        floor=0.50,
    )
    assert n == 2  # the two 0.52/0.55 ACCEPTs demoted; high ones kept
    assert [r.recommendation for r in rows[:5]] == [
        "REVIEW", "REVIEW", "ACCEPT", "ACCEPT", "ACCEPT"]


def test_apply_gate_noop_when_gap_at_floor():
    class R:
        def __init__(self, cov):
            self.recommendation, self.cov = "ACCEPT", cov
    rows = [R(0.58), R(0.66), R(0.74), R(0.85), R(0.99)]  # all clean, no sub-floor
    n = apply_calibrated_gate(
        rows,
        get_recommendation=lambda r: r.recommendation,
        set_review=lambda r, bar: setattr(r, "recommendation", "REVIEW"),
        get_coverage=lambda r: r.cov,
        floor=0.50,
    )
    assert n == 0
    assert all(r.recommendation == "ACCEPT" for r in rows)
