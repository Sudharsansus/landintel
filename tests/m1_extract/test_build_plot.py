"""Tests for Plot assembly against the real fixtures.

The headline check is the area cross-check: build a Plot in real-world metres
and confirm the geometry-derived area lands within tolerance of the FMB-stated
area on plots that close cleanly. That single number validates that scale was
applied correctly and uniformly. A plot whose boundary does not close is
required to come back honestly open, not force-closed into a fake area.

Runs the real M1 chain (vectors + OCR + anchor + build) -- no mocks. OCR is
slow, so all three plots are built once in a module-scoped fixture.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from landintel.core.enums import MeasurementSource, PlotStatus
from landintel.core.exceptions import GeometryError
from landintel.core.models import Plot
from landintel.pipeline.m1_extract.anchor import anchor_measurements
from landintel.pipeline.m1_extract.build_plot import (
    _drop_boundary_totals,
    build_plot,
    points_to_metres,
)
from landintel.pipeline.m1_extract.ocr import FmbHeader, extract_text, parse_header
from landintel.pipeline.m1_extract.pdf_vectors import PageVectors, Segment, extract_vectors

FMB_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "FMB"

AREA_TOLERANCE = 0.08  # 8%: clean fixtures land at 1.9% and 5.1%

CLIENT = "client_test"


def fmb_path(survey: int) -> Path:
    return FMB_DIR / f"FMB_SIVAGANGAI_Manamadurai_T.Pudukkottai_{survey}.pdf"


def build(survey: int) -> Plot:
    f = fmb_path(survey)
    vectors = extract_vectors(f)
    detections = extract_text(f)
    header = parse_header(detections)
    anchored = anchor_measurements(vectors, detections)
    return build_plot(
        client_id=CLIENT,
        vectors=vectors,
        detections=detections,
        anchor_result=anchored,
        header=header,
    )


@pytest.fixture(scope="module")
def plots() -> dict[int, Plot]:
    return {survey: build(survey) for survey in (100, 199, 31)}


# --- The area cross-check (scale validation) ---------------------------------


@pytest.mark.parametrize("survey", [100, 199, 31])
def test_computed_area_matches_stated_on_closed_plots(
    plots: dict[int, Plot], survey: int
) -> None:
    plot = plots[survey]
    assert plot.boundary is not None and plot.boundary.is_closed
    computed_ha = plot.boundary.computed_area / 10_000.0  # m^2 -> hectares
    assert plot.stated_area is not None
    error = abs(computed_ha - plot.stated_area) / plot.stated_area
    assert error <= AREA_TOLERANCE, (
        f"survey {survey}: computed {computed_ha:.3f} ha vs stated "
        f"{plot.stated_area:.3f} ha ({error:.1%})"
    )


# --- Honest non-closing boundary ---------------------------------------------


def test_open_geometry_stays_open_not_force_closed() -> None:
    """Segments that enclose no face -> open boundary, honestly, never faked shut.

    After the axis-aware frame fix, all 46 real fixtures close, so this exercises
    the open path with constructed geometry: an L of two edges that forms no
    ring. build_plot must represent it as open for the anomaly layer to flag,
    not force a closure that fabricates an area.
    """
    open_segments = [
        Segment(start=(0.0, 0.0), end=(100.0, 0.0), width=3.0),
        Segment(start=(100.0, 0.0), end=(100.0, 100.0), width=3.0),
    ]
    vectors = PageVectors(boundary=open_segments, page_width=595.0, page_height=841.0)
    header = FmbHeader(survey_no="X", district="D", taluk="T", village="V",
                       scale_denominator=2000, stated_area_ha=1.0)
    plot = build_plot(
        client_id=CLIENT,
        vectors=vectors,
        detections=[],
        anchor_result=anchor_measurements(vectors, []),
        header=header,
    )
    assert plot.boundary is not None
    assert plot.boundary.points, "an open boundary should still carry its points"
    assert plot.boundary.is_closed is False
    assert plot.boundary.closure_gap > 0.0


# --- Scale applied uniformly -------------------------------------------------


def test_scale_applied_uniformly_to_all_geometry(plots: dict[int, Plot]) -> None:
    """Corner points share the boundary's metre frame (one transform for all)."""
    plot = plots[100]
    assert plot.scale == 2021
    bx = [p[0] for p in plot.boundary.points]
    by = [p[1] for p in plot.boundary.points]
    # Same metre frame: corners share the boundary's coordinate range. A loose,
    # span-relative margin -- a corner stone may mark a point just outside the
    # polygonized perimeter; the point is that it is NOT in a different (pixel)
    # frame, which would be off by the ~0.71 m/pt scale factor.
    margin = 0.1 * max(max(bx) - min(bx), max(by) - min(by))
    for corner in plot.corner_points:
        assert min(bx) - margin <= corner.x <= max(bx) + margin
        assert min(by) - margin <= corner.y <= max(by) + margin


def test_geometry_is_in_metres_not_pixels(plots: dict[int, Plot]) -> None:
    """A village plot is tens of metres across and a fraction of a hectare-plus."""
    plot = plots[100]
    span_x = max(p[0] for p in plot.boundary.points) - min(p[0] for p in plot.boundary.points)
    assert 50.0 < span_x < 1000.0  # metres, not points (~215 m here)
    assert 100.0 < plot.boundary.computed_area < 1_000_000.0  # m^2


# --- Assembly contract -------------------------------------------------------


def test_metadata_carried_onto_plot(plots: dict[int, Plot]) -> None:
    plot = plots[100]
    assert plot.client_id == CLIENT
    assert plot.survey_no == "100"
    assert plot.district == "Sivagangai"
    assert plot.taluk == "Manamadurai"
    assert plot.village.startswith("T.Pudukkottai")
    assert plot.stated_area == pytest.approx(1.665)
    assert plot.status is PlotStatus.EXTRACTED


def test_measurements_unnormalized_with_confidence(plots: dict[int, Plot]) -> None:
    """build_plot does not normalize: values stay None, confidence carries."""
    plot = plots[100]
    assert plot.measurements, "expected anchored measurements on the plot"
    for m in plot.measurements:
        assert m.value is None  # normalization is the validator's job
        assert m.source is MeasurementSource.OCR
        assert 0.0 <= m.confidence <= 1.0
        # Chain dim measurements are routed directly (no anchor line available —
        # chain lines are excluded from candidates to avoid wrong snaps).
        if m.line_class != "chain":
            assert m.line_ref is not None


def test_all_measurements_render_upright(plots: dict[int, Plot]) -> None:
    """Every measurement rotation is in (-90, 90] so no label renders flipped.

    Regression: top/bottom edges have line angles near 176-178°, and a DXF text
    rotation in [90, 180) renders the label upside-down / mirror-flipped (e.g.
    "145,6" read as "9'St7L").  All dimension layers — boundary, internal and
    chain — must normalise to the upright range, matching the FMB sheet.
    """
    for survey, plot in plots.items():
        for m in plot.measurements:
            assert m.line_angle is not None, f"survey {survey}: {m.raw} has no angle"
            assert -90.0 < m.line_angle <= 90.0, (
                f"survey {survey}: {m.raw} angle {m.line_angle} renders flipped"
            )


def test_chain_dims_rotated_to_their_glyph_orientation(plots: dict[int, Plot]) -> None:
    """Chain dimension labels carry a non-flat rotation from the glyph PCA.

    Chain dims are not anchored to a line, so their rotation comes from the
    compound-fill outline's own orientation (det.angle_deg → _glyph_angle_to_dxf).
    The manual reference rotates them along their traverse; the regression here is
    that they must NOT all collapse to 0° (the old behaviour), and the rotation
    must be a readable, upright angle in (-90, 90].
    """
    chain = [m for m in plots[100].measurements if m.line_class == "chain"]
    assert len(chain) >= 8, "expected the survey-100 chain dimensions"
    # Every chain dim has an angle in the upright range.
    for m in chain:
        assert m.line_angle is not None
        assert -90.0 < m.line_angle <= 90.0
    # They are genuinely rotated, not a flat row of zeros: the diagonal traverse
    # legs (160.2 ≈ 27°, 93.8 ≈ -40°, 94.0 ≈ 80°) must show real tilt.
    by_val = {m.raw: m.line_angle for m in chain}
    assert abs(by_val["(160.2)"] - 27.0) < 6.0
    assert abs(by_val["(93.8)"] - (-40.0)) < 6.0
    assert abs(by_val["(94.0)"]) > 45.0  # steeply tilted, not horizontal


def test_chain_dims_placed_at_glyph_centre(plots: dict[int, Plot]) -> None:
    """Chain dim labels sit at the OCR glyph centre (PDF-exact placement)."""
    chain = [m for m in plots[100].measurements if m.line_class == "chain"]
    for m in chain:
        assert m.position is not None
        # in real-world metres, inside a sane page-derived range
        assert 0.0 < m.position[0] < 600.0
        assert 0.0 < m.position[1] < 700.0


def test_stone_snaps_to_junction_unit() -> None:
    """A stone glyph near a line junction snaps its coordinate onto it."""
    from landintel.pipeline.m1_extract.build_plot import _snap_stone
    from landintel.pipeline.m1_extract.pdf_vectors import Marker
    conn = [(100.0, 100.0), (200.0, 200.0)]
    m = Marker(x=105.0, y=100.0, width=4.0, height=6.0)  # 5 pt off the junction
    assert _snap_stone(m, conn) == (100.0, 100.0)  # snapped onto the junction


def test_stone_far_from_any_junction_is_not_snapped() -> None:
    """A stray red fill with no nearby junction keeps its own position."""
    from landintel.pipeline.m1_extract.build_plot import _snap_stone, _STONE_SNAP_MAX_PT
    from landintel.pipeline.m1_extract.pdf_vectors import Marker
    m = Marker(x=100.0 + _STONE_SNAP_MAX_PT + 10.0, y=100.0, width=4.0, height=6.0)
    assert _snap_stone(m, [(100.0, 100.0)]) == (m.x, m.y)  # too far -> left in place


def test_stone_coordinates_sit_on_boundary_or_subdivision_corners(
    plots: dict[int, Plot],
) -> None:
    """Every labelled stone's coordinate lands on a BOUNDARY/SUBDIVISION corner.

    Chain (traverse) endpoints are deliberately NOT in the candidate set: stones
    must snap to real survey corners, not to scattered dashed-traverse dash
    points.  Each stone coordinate must be ~0 m from a boundary/subdivision
    junction (its true corner, not the offset number-glyph position).
    """
    import math
    from landintel.pipeline.m1_extract.pdf_vectors import extract_vectors
    for survey in (100, 31):
        plot = plots[survey]
        v = extract_vectors(fmb_path(survey))
        ppm = points_to_metres(plot.scale)
        ph = v.page_height

        def tf(p: tuple[float, float]) -> tuple[float, float]:
            return (p[0] * ppm, (ph - p[1]) * ppm)

        # Boundary + subdivision corners ONLY (no chain).
        conn = list(plot.boundary.points)
        for s in list(v.internal) + list(v.boundary):
            conn.append(tf(s.start))
            conn.append(tf(s.end))
        for c in plot.corner_points:
            if not c.label:
                continue
            d = min(math.hypot(c.x - px, c.y - py) for px, py in conn)
            assert d < 0.5, f"survey {survey}: stone {c.label} is {d:.2f} m off any corner"


def test_chain_arrows_outside_boundary_are_dropped(plots: dict[int, Plot]) -> None:
    """Chain segments kept are all INSIDE the boundary (traverse arrows/neighbour
    legs outside are dropped as noise)."""
    from shapely.geometry import Polygon, Point as SP
    for survey in (100, 199):
        plot = plots[survey]
        poly = Polygon(plot.boundary.points).buffer(3.0 * points_to_metres(plot.scale))
        for a, b in plot.chain_segments:
            mid = SP((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
            assert poly.contains(mid), f"survey {survey}: chain segment outside boundary not dropped"


def test_simplify_collinear_merges_straight_only() -> None:
    """Collinear merge drops a straight-through vertex but keeps real corners."""
    from landintel.pipeline.m1_extract.build_plot import _simplify_collinear
    # Square with an extra MIDPOINT on the bottom edge (collinear) -> should drop it.
    ring = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
    out = _simplify_collinear(ring, tol_deg=2.0)
    assert (5.0, 0.0) not in out  # collinear midpoint removed
    assert len(out) - 1 == 4  # back to a 4-corner square
    # A real corner (90 deg) is never dropped.
    assert (10.0, 0.0) in out and (10.0, 10.0) in out


def _canon(text: str) -> str:
    return re.sub(r"[()\s]", "", text).replace(",", ".")


def test_boundary_edge_totals_are_dropped(plots: dict[int, Plot]) -> None:
    """The span TOTAL of a boundary edge (sum of its segments) is removed.

    Survey 100's bottom edges print both segment values and their sum:
    59.8 = 16.4+20.4+23.0 and 48.2 = 20.6+27.6.  Those two totals must be
    dropped (boundary lines only), while every segment value is kept.
    """
    bnd = {_canon(m.raw) for m in plots[100].measurements if m.line_class == "boundary"}
    # Totals removed.
    assert "59.8" not in bnd, "boundary total 59.8 should be dropped"
    assert "48.2" not in bnd, "boundary total 48.2 should be dropped"
    # Segments that sum to them are kept.
    for seg in ("16.4", "20.4", "23.0", "20.6", "27.6"):
        assert seg in bnd, f"segment {seg} must be kept"


def _mk_anchor(text: str, start, end, line_class: str = "boundary"):
    """Minimal stand-in for AnchoredMeasurement (text + line + class)."""
    return SimpleNamespace(
        text=text, line_class=line_class,
        line=SimpleNamespace(start=start, end=end),
    )


def test_drop_boundary_totals_removes_span_sum() -> None:
    """A boundary value equal to the sum of its collinear segments is dropped."""
    # Edge along x: segments 20 (0->20) and 30 (20->50); the 50 total is anchored
    # to a short stub (0->10) so value(50) >> line(10) and 50 == 20+30.
    segs = [
        _mk_anchor("20.0", (0.0, 0.0), (20.0, 0.0)),
        _mk_anchor("30.0", (20.0, 0.0), (50.0, 0.0)),
        _mk_anchor("50.0", (0.0, 0.0), (10.0, 0.0)),
    ]
    kept = {a.text for a in _drop_boundary_totals(segs, ppm=1.0)}
    assert kept == {"20.0", "30.0"}, "the span total 50.0 must be dropped"


def test_drop_keeps_real_long_edge() -> None:
    """A long edge whose value is NOT a sum of collinear neighbours is kept."""
    segs = [
        _mk_anchor("20.0", (0.0, 0.0), (20.0, 0.0)),
        _mk_anchor("30.0", (20.0, 0.0), (50.0, 0.0)),
        _mk_anchor("45.0", (0.0, 0.0), (10.0, 0.0)),  # 45 != 20+30 -> not a total
    ]
    kept = {a.text for a in _drop_boundary_totals(segs, ppm=1.0)}
    assert "45.0" in kept, "a non-summing long edge must NOT be dropped"


def test_drop_is_boundary_class_only() -> None:
    """Internal totals are NOT dropped — only boundary lines are affected."""
    segs = [
        _mk_anchor("20.0", (0.0, 0.0), (20.0, 0.0), "internal"),
        _mk_anchor("30.0", (20.0, 0.0), (50.0, 0.0), "internal"),
        _mk_anchor("50.0", (0.0, 0.0), (10.0, 0.0), "internal"),
    ]
    kept = {a.text for a in _drop_boundary_totals(segs, ppm=1.0)}
    assert "50.0" in kept, "internal-line totals must be kept (boundary-only rule)"


def test_missing_scale_is_a_hard_error() -> None:
    """No scale -> GeometryError, never silent pixel-unit geometry."""
    header = FmbHeader(survey_no="999", district="D", taluk="T", village="V",
                       scale_denominator=None, stated_area_ha=1.0)
    with pytest.raises(GeometryError):
        build_plot(
            client_id=CLIENT,
            vectors=PageVectors(page_width=595.0, page_height=841.0),
            detections=[],
            anchor_result=anchor_measurements(PageVectors(), []),
            header=header,
        )


def test_points_to_metres_factor() -> None:
    """The conversion factor is the documented point->metre chain."""
    assert points_to_metres(2021) == pytest.approx((2.54 / 72.0) * 2021 / 100.0)
