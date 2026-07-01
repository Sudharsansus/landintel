"""Tests for the clubbed-M2 QA overlay (m2_club.qa_render).

Renders a small synthetic clubbed set to a PNG -- headless (Agg), no real data --
with and without a cadastral_source, and confirms the misplacement path runs.
"""
from __future__ import annotations

import numpy as np

from landintel.pipeline.m2_club import club_pipeline
from landintel.pipeline.m2_club.qa_render import misplaced, render_club_qa

from conftest import BASE_X, BASE_Y, RECT, MockCadastral, build_fmb, utm_polygon

CRS = "EPSG:32643"
RECT_B = [("A", 50.0, 0.0), ("B", 90.0, 0.0), ("C", 90.0, 30.0), ("D", 50.0, 30.0)]


def _clubbed(tmp_path):
    a = build_fmb(tmp_path / "m1_100.dxf", "100", RECT)
    b = build_fmb(tmp_path / "m1_101.dxf", "101", RECT_B)
    src = MockCadastral({"100": utm_polygon(RECT), "101": utm_polygon(RECT_B)})
    results = club_pipeline([a, b], tmp_path / "out", crs=CRS, cadastral_source=src)
    return results, src


def _nonempty_png(path) -> bool:
    return path.exists() and path.stat().st_size > 0


def test_render_without_cadastral(tmp_path):
    results, _src = _clubbed(tmp_path)
    out = tmp_path / "qa_no_cad.png"
    ret = render_club_qa(results, out, crs=CRS)
    assert ret == out
    assert _nonempty_png(out)


def test_render_with_cadastral(tmp_path):
    results, src = _clubbed(tmp_path)
    out = tmp_path / "qa_cad.png"
    ret = render_club_qa(results, out, cadastral_source=src, crs=CRS)
    assert ret == out
    assert _nonempty_png(out)


def test_render_flags_misplacement(tmp_path):
    """A placed footprint far from its parcel triggers the MISPLACED path (still
    renders a valid PNG)."""
    results, _src = _clubbed(tmp_path)
    by = {r.survey_number: r for r in results}
    # cadastre puts 100's parcel far (>100 m) from where it's placed.
    far_src = MockCadastral({
        "100": utm_polygon(RECT, tx=BASE_X + 5000.0, ty=BASE_Y + 5000.0),
        "101": utm_polygon(RECT_B),
    })
    out = tmp_path / "qa_misplaced.png"
    ret = render_club_qa(results, out, cadastral_source=far_src, crs=CRS)
    assert ret == out and _nonempty_png(out)
    # sanity on the helper used internally.
    flag, d = misplaced((0.0, 0.0), (200.0, 0.0), 100.0)
    assert flag and abs(d - 200.0) < 1e-6
    # 100 is placed, so its centroid is far from the shifted parcel -> would flag.
    assert by["100"].placed


def test_render_empty_results(tmp_path):
    """No placements at all still produces a valid (non-empty) PNG."""
    out = tmp_path / "qa_empty.png"
    ret = render_club_qa([], out, crs=CRS)
    assert ret == out and _nonempty_png(out)
