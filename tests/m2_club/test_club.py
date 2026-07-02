"""Tests for the new M2 (m2_club): georeference FMB DXFs WITHOUT a surveyor file,
using all methods, cross-checked, 0 false positives.
"""
from __future__ import annotations

import numpy as np

from landintel.pipeline.m2_club import cadastral_seat, club_pipeline, gps_seat
from landintel.pipeline.m2_club import relative_club as RC
from landintel.pipeline.m2_georef.extract_m1 import extract_m1_dxf

from conftest import BASE_X, BASE_Y, RECT, MockCadastral, build_fmb, utm_polygon


# --------------------------------------------------------------------------
# Method 1: cadastral seat (survey# -> parcel), strict rigid shape gate.
# --------------------------------------------------------------------------

def test_cadastral_seat_passes_gate_on_correct_parcel(tmp_path):
    dxf = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    m1 = extract_m1_dxf(dxf)
    src = MockCadastral({"100": utm_polygon(RECT)})

    cand = cadastral_seat(m1, src)
    assert cand is not None and cand.method == "cadastral"
    assert cand.passes_gate, cand.note
    assert 0.9 <= cand.area_ratio <= 1.1
    cx, cy = cand.centroid()
    assert abs(cx - (BASE_X + 25)) < 5 and abs(cy - (BASE_Y + 15)) < 5


def test_cadastral_seat_wrong_size_below_gate(tmp_path):
    """A right-survey-number but wrong-SIZE parcel must NOT pass the gate (the
    scale/area gate is the 0-FP arbiter: identity collisions stay REVIEW)."""
    dxf = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    m1 = extract_m1_dxf(dxf)
    src = MockCadastral({"100": utm_polygon(RECT, scale=1.6)})  # 60% too big

    cand = cadastral_seat(m1, src)
    assert cand is not None
    assert not cand.passes_gate          # below gate -> pipeline will mark REVIEW


def test_cadastral_seat_no_parcel_returns_none(tmp_path):
    dxf = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    m1 = extract_m1_dxf(dxf)
    assert cadastral_seat(m1, MockCadastral({})) is None


def test_cadastral_seat_off_seat_label_demotes(tmp_path):
    """Seat-locality 0-FP lock: a right-SHAPE parcel whose LABEL POINT is far from where
    the plot seats is a wrong-parcel collision -> below gate (REVIEW), never ACCEPT."""
    dxf = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    m1 = extract_m1_dxf(dxf)

    class _OffSeatSrc(MockCadastral):
        def label_point(self, sn):                  # label 700 m from the (correct) parcel
            return (BASE_X + 500.0, BASE_Y + 500.0)

    cand = cadastral_seat(m1, _OffSeatSrc({"100": utm_polygon(RECT)}))
    assert cand is not None
    assert not cand.passes_gate and "off-seat" in cand.note


# --------------------------------------------------------------------------
# Method 2: GPS / control-point seat (>=3-point LSQ similarity + seed quality).
# --------------------------------------------------------------------------

def test_gps_seat_places_corners_on_control(tmp_path):
    dxf = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    m1 = extract_m1_dxf(dxf)
    # Client directive 2026-07-02: minimum 3 stone matches per FMB for full quality.
    control = [("A", (BASE_X, BASE_Y)), ("B", (BASE_X + 50.0, BASE_Y)),
               ("C", (BASE_X + 50.0, BASE_Y + 30.0))]

    cand = gps_seat(m1, control)
    assert cand is not None and cand.method == "gps_seed"
    assert cand.seed_ok and cand.passes_gate          # 3 points, adequate baseline
    idx = {s.label: s.index for s in m1.stones}
    assert np.allclose(cand.adjusted[idx["A"]], [BASE_X, BASE_Y], atol=0.5)
    assert np.allclose(cand.adjusted[idx["B"]], [BASE_X + 50, BASE_Y], atol=0.5)
    assert np.allclose(cand.adjusted[idx["C"]], [BASE_X + 50, BASE_Y + 30], atol=0.5)


def test_gps_seat_two_points_is_review(tmp_path):
    """A 2-point seat has no redundancy: still placed, but demoted to REVIEW
    (seed_ok=False) -- min 3 stone matches for ACCEPT_SEEDED."""
    dxf = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    m1 = extract_m1_dxf(dxf)
    control = [("A", (BASE_X, BASE_Y)), ("B", (BASE_X + 50.0, BASE_Y))]

    cand = gps_seat(m1, control)
    assert cand is not None and cand.method == "gps_seed"
    assert not cand.seed_ok and not cand.passes_gate
    assert "control points" in cand.note
    # geometry itself is still the exact 2-point placement (position is right)
    idx = {s.label: s.index for s in m1.stones}
    assert np.allclose(cand.adjusted[idx["A"]], [BASE_X, BASE_Y], atol=0.5)


def test_gps_seat_bad_control_point_is_review(tmp_path):
    """3 points where one disagrees with the fit by >2 m -> REVIEW (LSQ residual
    gate catches bad GPS / mislabels that a 2-point fit would swallow silently)."""
    dxf = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    m1 = extract_m1_dxf(dxf)
    control = [("A", (BASE_X, BASE_Y)), ("B", (BASE_X + 50.0, BASE_Y)),
               ("C", (BASE_X + 50.0 + 8.0, BASE_Y + 30.0 + 8.0))]  # ~11 m off

    cand = gps_seat(m1, control)
    assert cand is not None
    assert not cand.seed_ok and not cand.passes_gate
    assert "residual" in cand.note


def test_gps_seat_short_baseline_not_ok(tmp_path):
    tiny = [("A", 0.0, 0.0), ("B", 3.0, 0.0), ("C", 3.0, 2.0), ("D", 0.0, 2.0)]
    dxf = build_fmb(tmp_path / "m1_tiny.dxf", "200", tiny)
    m1 = extract_m1_dxf(dxf)
    control = [("A", (BASE_X, BASE_Y)), ("B", (BASE_X + 3.0, BASE_Y)),
               ("C", (BASE_X + 3.0, BASE_Y + 2.0))]  # 3 points but ~3.6 m baseline

    cand = gps_seat(m1, control)
    assert cand is not None
    assert not cand.seed_ok and not cand.passes_gate  # short baseline -> REVIEW


# --------------------------------------------------------------------------
# Method 3: relative clubbing (label-free corroboration + gated propagation).
# --------------------------------------------------------------------------

def _seat_via_gps(tmp_path, name, survey, corners, neighbors, ax, ay):
    dxf = build_fmb(tmp_path / name, survey, corners, neighbors)
    m1 = extract_m1_dxf(dxf)
    # two corners of the relative frame -> known UTM (angle 0, scale 1)
    c0 = corners[0]; c1 = corners[1]
    control = [(c0[0], (ax + c0[1], ay + c0[2])), (c1[0], (ax + c1[1], ay + c1[2]))]
    return m1, gps_seat(m1, control)


def test_shares_edge_and_corroboration(tmp_path):
    # A at base; B sharing A's east edge (x=50), both seated by GPS.
    rect_b = [("A", 50.0, 0.0), ("B", 90.0, 0.0), ("C", 90.0, 30.0), ("D", 50.0, 30.0)]
    m1a, pa = _seat_via_gps(tmp_path, "a.dxf", "100", RECT, None, BASE_X, BASE_Y)
    m1b, pb = _seat_via_gps(tmp_path, "b.dxf", "101", rect_b, None, BASE_X, BASE_Y)
    assert pa is not None and pb is not None

    assert RC.shares_edge(pa, pb)        # they meet on the x=719650 edge
    corro = RC.corroborate_seated({"100": pa, "101": pb}, {"100": m1a, "101": m1b})
    assert "101" in corro["100"] and "100" in corro["101"]


def test_propagate_from_seated_tiles_neighbor(tmp_path):
    # A seated by GPS at base; B has NO seat, only a neighbour label naming A.
    m1a, pa = _seat_via_gps(
        tmp_path, "a.dxf", "100", RECT, [("101", (52.0, 15.0))], BASE_X, BASE_Y)
    rect_b = [("P", 0.0, 0.0), ("Q", 40.0, 0.0), ("R", 40.0, 30.0), ("S", 0.0, 30.0)]
    dxf_b = build_fmb(tmp_path / "b.dxf", "101", rect_b, [("100", (-2.0, 15.0))])
    m1b = extract_m1_dxf(dxf_b)

    prop = RC.propagate_from_seated(m1b, pa, m1a, [pa.footprint()])
    assert prop is not None and prop.method == "propagated"
    fb, fa = prop.footprint(), pa.footprint()
    assert fb is not None
    # tiles A: touches but interiors do not overlap.
    inter = fb.intersection(fa).area / min(fb.area, fa.area)
    assert inter < 0.12


# --------------------------------------------------------------------------
# End-to-end pipeline: club FMBs into one georeferenced file.
# --------------------------------------------------------------------------

def test_pipeline_clubs_cadastral_plots(tmp_path):
    a = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    rect_b = [("A", 50.0, 0.0), ("B", 90.0, 0.0), ("C", 90.0, 30.0), ("D", 50.0, 30.0)]
    b = build_fmb(tmp_path / "m1_101.dxf", "101", rect_b)
    src = MockCadastral({"100": utm_polygon(RECT), "101": utm_polygon(rect_b)})

    out = tmp_path / "out"
    results = club_pipeline([a, b], out, cadastral_source=src)
    by = {r.survey_number: r for r in results}
    assert by["100"].placed and by["101"].placed
    assert (out / "clubbed_village.dxf").exists()
    assert (out / "clubbed.geojson").exists()
    assert (out / "clubbed_points.csv").exists()


def test_pipeline_propagates_neighbor_without_parcel(tmp_path):
    a = build_fmb(tmp_path / "m1_100.dxf", "100", RECT, [("101", (52.0, 15.0))])
    rect_b = [("P", 0.0, 0.0), ("Q", 40.0, 0.0), ("R", 40.0, 30.0), ("S", 0.0, 30.0)]
    b = build_fmb(tmp_path / "m1_101.dxf", "101", rect_b, [("100", (-2.0, 15.0))])
    # A gets an absolute seat from GPS (3 control points = full quality);
    # B has neither parcel nor GPS.
    gps = {"100": [("A", (BASE_X, BASE_Y)), ("B", (BASE_X + 50.0, BASE_Y)),
                   ("C", (BASE_X + 50.0, BASE_Y + 30.0))]}

    results = club_pipeline([a, b], tmp_path / "out", gps_control=gps)
    by = {r.survey_number: r for r in results}
    assert by["100"].recommendation == "ACCEPT_SEEDED"
    assert by["101"].recommendation == "ACCEPT" and by["101"].method == "propagated"


def test_pipeline_wrong_size_parcel_is_review_not_accept(tmp_path):
    """0-FP: an oversized same-number parcel is located but NOT accepted."""
    a = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    src = MockCadastral({"100": utm_polygon(RECT, scale=1.6)})
    results = club_pipeline([a], tmp_path / "out", cadastral_source=src)
    assert results[0].recommendation == "REVIEW"
    assert not results[0].placed


def test_pipeline_no_method_is_no_coverage(tmp_path):
    """A plot with no parcel, no GPS, no seated neighbour is honestly NO_COVERAGE
    (staged to scale, never guessed onto a position)."""
    a = build_fmb(tmp_path / "m1_999.dxf", "999", RECT)
    results = club_pipeline([a], tmp_path / "out", cadastral_source=MockCadastral({}))
    assert results[0].recommendation == "NO_COVERAGE"
    assert not results[0].placed
