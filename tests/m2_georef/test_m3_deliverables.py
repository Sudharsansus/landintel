"""M3 stone-cloud finishing: scale-lock + honest disposition cascade + deliverables.

Synthetic only -- no village fixtures, no live geocode. Proves rule 2 (placement emits
scale 1 even when the true scale differs, with the fitted scale kept as a diagnostic),
the first-class disposition cascade, and that the three deliverables write.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from landintel.pipeline.m2_georef.m3_deliverables import (
    M3Placement, classify, place_scale_locked, write_dxf, write_overlay, write_report)


# ------------------------------------------------------- scale-locked placement ----
def test_place_emits_scale_one_but_reports_fitted_scale():
    # dst is src rotated 30 deg, translated, AND scaled 1.05 -> a rigid (s=1) fit CANNOT
    # zero the residuals, and the diagnostic s_fitted must reveal the ~1.05 scale.
    src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    th = np.deg2rad(30.0)
    Rt = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = (1.05 * src @ Rt.T) + np.array([500.0, 700.0])

    R, t, s_fitted, residuals = place_scale_locked(src, dst)

    # rule 2: the emitted rotation is orthonormal (scale 1), det +1 (no reflection)
    assert np.allclose(R @ R.T, np.eye(2), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    # the DIAGNOSTIC scale recovered the 1.05 the rigid fit refused to apply
    assert s_fitted == pytest.approx(1.05, abs=0.02)
    # a 5% scale the rigid fit did not absorb shows up as a non-zero residual
    assert float(np.median(residuals)) > 0.1


def test_place_recovers_exact_rigid_transform():
    # No scale change -> a rigid fit is exact, residuals ~0, s_fitted ~1.
    src = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    th = np.deg2rad(-42.0)
    Rt = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = (src @ Rt.T) + np.array([123.0, 456.0])
    R, t, s_fitted, residuals = place_scale_locked(src, dst)
    assert float(np.max(residuals)) < 1e-6
    assert s_fitted == pytest.approx(1.0, abs=1e-6)
    assert np.allclose((src @ R.T + t), dst, atol=1e-6)


# ---------------------------------------------------------- disposition cascade ----
def test_classify_accept_survey_grade():
    d, note = classify(n_matched=5, n_corners=5, median_resid=0.8, max_resid=1.5,
                       s_fitted=1.0, tiles=True, window_has_stones=True)
    assert d == "ACCEPT" and "survey-grade" in note


def test_classify_review_when_median_over_survey_grade():
    # located (5 stones) but median 2.6 m > 2.0 accept bound -> REVIEW, not ACCEPT
    d, _ = classify(5, 5, median_resid=2.6, max_resid=3.2, s_fitted=1.0,
                    tiles=True, window_has_stones=True)
    assert d == "REVIEW"


def test_classify_review_when_overlaps():
    # a good fit that does NOT tile is REVIEW (confirm extent), never ACCEPT
    d, note = classify(6, 6, median_resid=0.5, max_resid=1.0, s_fitted=1.0,
                       tiles=False, window_has_stones=True)
    assert d == "REVIEW" and "overlaps" in note


def test_classify_accept_bar_is_data_keyed_to_corner_count():
    # a 4-corner plot cannot be asked for 5 stones -> full match of its 4 corners ACCEPTs
    d, _ = classify(n_matched=4, n_corners=4, median_resid=0.7, max_resid=1.1,
                    s_fitted=1.0, tiles=True, window_has_stones=True)
    assert d == "ACCEPT"


def test_classify_needs_gps_when_stones_present_but_fit_weak():
    d, note = classify(0, 6, median_resid=float("nan"), max_resid=float("nan"),
                       s_fitted=float("nan"), tiles=True, window_has_stones=True)
    assert d == "NEEDS_GPS" and "needs GPS" in note


def test_classify_unmeasured_when_no_stones_in_window():
    d, note = classify(0, 6, float("nan"), float("nan"), float("nan"),
                       tiles=True, window_has_stones=False)
    assert d == "UNMEASURED" and "data gap" in note


def test_classify_scale_out_of_band_is_not_accept():
    # A diagnostic scale outside the VALIDATED 0.5-2.0 band signals an M1 unit bug ->
    # never survey-grade ACCEPT. (The band is the coarse sanity net; the median-residual
    # gate is the primary protector -- a truly mis-scaled fit also blows the residual.)
    d, _ = classify(6, 6, median_resid=0.4, max_resid=0.9, s_fitted=2.5,
                    tiles=True, window_has_stones=True)
    assert d == "REVIEW"


# ------------------------------------------------------------------ deliverables ----
def _placement(sv, disp, cx, cy):
    ring = np.array([[cx, cy], [cx + 20, cy], [cx + 20, cy + 20], [cx, cy + 20]], float)
    return M3Placement(survey_number=sv, disposition=disp, R=np.eye(2),
                       t=np.array([cx, cy]), s_fitted=1.0, ring_utm=ring,
                       n_matched=5, n_corners=4, median_residual_m=0.7, max_residual_m=1.1)


def test_deliverables_write(tmp_path):
    placements = [
        _placement("100", "ACCEPT", 0.0, 0.0),
        _placement("101", "REVIEW", 100.0, 100.0),
        M3Placement(survey_number="102", disposition="NEEDS_GPS", n_corners=5),
    ]
    stones = np.array([[5.0, 5.0], [15.0, 15.0], [105.0, 105.0]], float)

    dxf = write_dxf(placements, tmp_path / "clubbed_village.dxf")
    png = write_overlay(placements, stones, tmp_path / "qa_overlay.png", village="SYNTH")
    rep = write_report(placements, tmp_path / "m3_report.json", village="SYNTH")

    assert dxf.exists() and png.exists() and rep.exists()
    data = json.loads(rep.read_text())
    assert data["disposition_counts"] == {"ACCEPT": 1, "REVIEW": 1, "NEEDS_GPS": 1}
    # report is ordered ACCEPT-first and carries the scale-lock evidence
    assert data["plots"][0]["survey_number"] == "100"
    assert data["plots"][0]["scale_locked_to"] == 1.0
    # the NEEDS_GPS plot has no placement geometry and is not in the DXF ring set
    ng = next(p for p in data["plots"] if p["survey_number"] == "102")
    assert ng["median_residual_m"] is None
