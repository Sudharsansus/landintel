"""Tests for the clubbed-M2 verification suite (m2_club.verify).

The clean clubbed set passes every check. Each injected defect (overlap, oversized,
empty/no placement) fails the RIGHT check, and the demote-only gate turns an ACCEPT
that fails a HARD geometry check into REVIEW -- never the reverse (0-FP discipline).
"""
from __future__ import annotations

import numpy as np

from landintel.pipeline.m2_club import club_pipeline
from landintel.pipeline.m2_club.placement import CandidatePlacement, ClubResult
from landintel.pipeline.m2_club.verify import (
    ClubVerifyResult,
    gate_results,
    verify_club,
    verify_placed_plot,
)

from conftest import BASE_X, BASE_Y, RECT, MockCadastral, build_fmb, utm_polygon

CRS = "EPSG:32643"

RECT_B = [("A", 50.0, 0.0), ("B", 90.0, 0.0), ("C", 90.0, 30.0), ("D", 50.0, 30.0)]


def _clean_clubbed(tmp_path):
    """Two adjacent cadastral-seated plots -> a clean, tiling clubbed set."""
    a = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    b = build_fmb(tmp_path / "m1_101.dxf", "101", RECT_B)
    src = MockCadastral({"100": utm_polygon(RECT), "101": utm_polygon(RECT_B)})
    return club_pipeline([a, b], tmp_path / "out", crs=CRS, cadastral_source=src)


# --------------------------------------------------------------------------
# Clean set passes everything.
# --------------------------------------------------------------------------

def test_clean_clubbed_set_passes_all_checks(tmp_path):
    results = _clean_clubbed(tmp_path)
    vr = verify_club(results, CRS)
    assert isinstance(vr, ClubVerifyResult)
    assert vr.all_passed, vr.failed_names()
    # all six named checks present.
    names = {c.name for c in vr.checks}
    assert names == {
        "CLOSURE_AREA", "UTM_RANGE", "RIGID_SCALE",
        "STONE_COUNT_PRESERVED", "NON_OVERLAPPING_TILING", "ACCOUNTED",
    }


def test_pipeline_writes_verify_sidecar(tmp_path):
    out = tmp_path / "out"
    _ = _clean_clubbed(tmp_path)
    assert (out / "clubbed.verify.txt").exists()
    txt = (out / "clubbed.verify.txt").read_text(encoding="utf-8")
    assert "CLUB VERIFY" in txt and "CLOSURE_AREA" in txt


# --------------------------------------------------------------------------
# Injected defects fail the right check.
# --------------------------------------------------------------------------

def test_overlapping_accepts_fail_tiling(tmp_path):
    results = _clean_clubbed(tmp_path)
    by = {r.survey_number: r for r in results}
    # Move plot 101 to sit on top of 100 -> interiors overlap.
    p = by["101"].placement
    p.adjusted = by["100"].placement.adjusted.copy()
    vr = verify_club(results, CRS)
    assert not vr.get("NON_OVERLAPPING_TILING").passed
    # the geometry of each plot is still individually fine.
    assert vr.get("CLOSURE_AREA").passed
    assert vr.get("UTM_RANGE").passed


def test_oversized_scale_fails_rigid_scale(tmp_path):
    results = _clean_clubbed(tmp_path)
    by = {r.survey_number: r for r in results}
    by["100"].placement.scale = 1.6        # outside [0.8, 1.25]
    vr = verify_club(results, CRS)
    assert not vr.get("RIGID_SCALE").passed


def test_out_of_range_fails_utm_range(tmp_path):
    results = _clean_clubbed(tmp_path)
    by = {r.survey_number: r for r in results}
    # shove the placement back near the relative-metre origin (a gross mis-georef).
    by["100"].placement.adjusted = by["100"].placement.adjusted - np.array(
        [BASE_X, BASE_Y])
    vr = verify_club(results, CRS)
    assert not vr.get("UTM_RANGE").passed


def test_empty_placement_fails_closure(tmp_path):
    results = _clean_clubbed(tmp_path)
    by = {r.survey_number: r for r in results}
    # a placement whose ring collapses to < 3 distinct points -> no footprint.
    p = by["100"].placement
    p.adjusted = np.tile(p.adjusted[0], (len(p.adjusted), 1))
    vr = verify_club(results, CRS)
    assert not vr.get("CLOSURE_AREA").passed


def test_stone_count_dropped_fails_count(tmp_path):
    results = _clean_clubbed(tmp_path)
    by = {r.survey_number: r for r in results}
    # drop a placed stone -> placed count no longer matches the source M1 DXF.
    by["100"].placement.adjusted = by["100"].placement.adjusted[:-1]
    vr = verify_club(results, CRS)
    assert not vr.get("STONE_COUNT_PRESERVED").passed


# --------------------------------------------------------------------------
# ACCOUNTED + per-plot helper.
# --------------------------------------------------------------------------

def test_accounted_check(tmp_path):
    results = _clean_clubbed(tmp_path)
    # all clean -> accounted.
    assert verify_club(results, CRS).get("ACCOUNTED").passed
    # corrupt one disposition to an invalid state.
    results[0].recommendation = "BOGUS"
    vr = verify_club(results, CRS)
    assert not vr.get("ACCOUNTED").passed


def test_verify_placed_plot_reports_failing_checks(tmp_path):
    results = _clean_clubbed(tmp_path)
    by = {r.survey_number: r for r in results}
    assert verify_placed_plot(by["100"], CRS) == []      # clean
    by["100"].placement.scale = 2.0
    assert "RIGID_SCALE" in verify_placed_plot(by["100"], CRS)


def test_verify_placed_plot_no_placement(tmp_path):
    r = ClubResult(m1_file="nope.dxf", survey_number="9", recommendation="REVIEW")
    assert "CLOSURE_AREA" in verify_placed_plot(r, CRS)


# --------------------------------------------------------------------------
# DEMOTE-ONLY gate.
# --------------------------------------------------------------------------

def test_gate_demotes_accept_failing_hard_check(tmp_path):
    results = _clean_clubbed(tmp_path)
    by = {r.survey_number: r for r in results}
    assert by["100"].recommendation == "ACCEPT"
    # inject a hard failure (scale warp) on an ACCEPT.
    by["100"].placement.scale = 1.9
    vr = gate_results(results, CRS)
    assert by["100"].recommendation == "REVIEW"           # demoted
    assert "demoted by verify" in by["100"].note
    # 101 untouched and still ACCEPT.
    assert by["101"].recommendation == "ACCEPT"
    # after demotion, RIGID_SCALE check no longer sees a placed warp.
    assert vr.get("RIGID_SCALE").passed


def test_gate_never_promotes(tmp_path):
    """A REVIEW/NO_COVERAGE is never lifted by the gate, even if geometry is clean."""
    a = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    src = MockCadastral({"100": utm_polygon(RECT, scale=1.6)})  # oversized -> REVIEW
    results = club_pipeline([a], tmp_path / "out", crs=CRS, cadastral_source=src)
    assert results[0].recommendation == "REVIEW"
    gate_results(results, CRS)
    assert results[0].recommendation == "REVIEW"          # stays REVIEW, no promotion


def test_gate_demotes_only_hard_not_overlap(tmp_path):
    """An overlap (a SET-level check) is reported but is NOT a per-plot demotion
    trigger -- the pipeline's conflict resolver handles overlaps; gate_results only
    demotes on hard per-plot geometry failures."""
    results = _clean_clubbed(tmp_path)
    by = {r.survey_number: r for r in results}
    # Both individually-clean but overlapping (no hard per-plot failure).
    by["101"].placement.adjusted = by["100"].placement.adjusted.copy()
    gate_results(results, CRS)
    # Neither is demoted by gate_results (overlap is not a hard per-plot check).
    assert by["100"].recommendation == "ACCEPT"
    assert by["101"].recommendation == "ACCEPT"
