"""Round-trip tests for DXF serialization.

The contract is layer fidelity and losslessness: write a Plot, read the DXF back
with ezdxf, and confirm every canonical layer exists, each element sits on its
correct layer, and the boundary geometry/area survives the round-trip.

A fast synthetic Plot covers the serialization logic deterministically; one real
fixture (survey 100, via the full M1 chain) confirms end-to-end fidelity.
"""

from __future__ import annotations

from pathlib import Path

import ezdxf
import pytest
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

from landintel.core.enums import LayerType
from landintel.core.models import (
    Boundary,
    CornerPoint,
    Measurement,
    NeighborLabel,
    Plot,
    SubPlot,
    SubPlotLabel,
)
from landintel.pipeline.m1_extract.anchor import anchor_measurements
from landintel.pipeline.m1_extract.build_plot import build_plot
from landintel.pipeline.m1_extract.ocr import extract_text, parse_header
from landintel.pipeline.m1_extract.pdf_vectors import extract_vectors
from landintel.pipeline.m1_extract.to_dxf import build_document, text_height_for, write_dxf

FMB_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "FMB"


def by_layer(msp, layer: LayerType) -> list:
    return [e for e in msp if e.dxf.layer == layer.value]


@pytest.fixture
def synthetic_plot() -> Plot:
    """A small, fully-specified plot exercising every populated layer."""
    return Plot(
        client_id="client_test",
        survey_no="42",
        district="D",
        taluk="T",
        village="V",
        scale=2000,
        stated_area=0.01,
        boundary=Boundary(points=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]),
        corner_points=[
            CornerPoint(label="A", x=0.0, y=0.0),
            CornerPoint(label="", x=10.0, y=10.0),  # unlabelled -> synthetic ?N label
        ],
        measurements=[
            Measurement(raw="10.0", line_class="boundary", position=(5.0, 0.0)),
            Measurement(raw="(7.1)", line_class="chain", position=(5.0, 5.0)),
            Measurement(raw="3.2", line_class="internal", position=(2.0, 2.0)),
        ],
    )


# --- Layer fidelity ----------------------------------------------------------


def test_all_canonical_layers_exist(synthetic_plot: Plot) -> None:
    doc = build_document(synthetic_plot)
    for layer in LayerType:
        assert layer.value in doc.layers, f"missing layer {layer.value}"


def test_elements_written_to_correct_layers(synthetic_plot: Plot) -> None:
    msp = build_document(synthetic_plot).modelspace()

    boundary = by_layer(msp, LayerType.BOUNDARY)
    # synthetic_plot: 5 points (0,0)→(10,0)→(10,10)→(0,10)→(0,0) = 4 edges.
    assert len(boundary) == 4
    assert all(e.dxftype() == "LWPOLYLINE" for e in boundary)

    survey = by_layer(msp, LayerType.SURVEY_NUMBER)
    assert len(survey) == 1 and survey[0].dxf.text == "42"

    # EVERY corner stone is written as TEXT (position is load-bearing for M2);
    # an unlabelled corner gets a synthetic "?N" placeholder, never dropped.
    stones = by_layer(msp, LayerType.STONES)
    assert len(stones) == len(synthetic_plot.corner_points) == 2
    assert all(s.dxftype() == "TEXT" for s in stones)
    texts = {s.dxf.text for s in stones}
    assert "A" in texts                       # the labelled corner
    assert any(t.startswith("?") for t in texts)  # the synthetic placeholder
    # Stone text is LEFT-justified (halign=Left/valign=Baseline) and its insertion
    # point (group 10 — the grip) sits EXACTLY on the coordinate (connecting
    # point), so it joins the line-endpoint node in AutoCAD with no manual step.
    assert stones[0].dxf.halign == 0  # Left
    assert stones[0].dxf.valign == 0  # Baseline
    ins = stones[0].dxf.insert
    assert (round(ins.x, 6), round(ins.y, 6)) == (0.0, 0.0)  # CornerPoint A is at (0,0)

    # Dimensions routed by line class; dot->comma for boundary AND chain dims.
    bdim = by_layer(msp, LayerType.BOUNDARY_DIMENSIONS)
    cdim = by_layer(msp, LayerType.CHAINLINE_DIMENSIONS)
    gdim = by_layer(msp, LayerType.DIMENSIONS)
    assert [e.dxf.text for e in bdim] == ["10,0"]  # dot->comma per client convention
    assert [e.dxf.text for e in cdim] == ["(7,1)"]  # dot->comma applied to chain dims too
    assert [e.dxf.text for e in gdim] == ["3,2"]  # dot->comma now applied to all dimension layers


def test_unpopulated_layers_are_empty_but_present(synthetic_plot: Plot) -> None:
    """Raw-line layers exist (convention) but carry nothing for a semantic Plot."""
    msp = build_document(synthetic_plot).modelspace()
    for layer in (LayerType.CHAIN_LINES, LayerType.SUBDIVISION_LINES, LayerType.BLUE_STROKES):
        assert by_layer(msp, layer) == []


# --- Lossless round-trip -----------------------------------------------------


def test_roundtrip_preserves_boundary_geometry(synthetic_plot: Plot, tmp_path: Path) -> None:
    path = write_dxf(synthetic_plot, tmp_path / "plot42.dxf")
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    boundary = by_layer(msp, LayerType.BOUNDARY)
    assert len(boundary) == 4  # 4 edges for the 10x10 square

    # Every original vertex appears as a start or end point of some LWPOLYLINE edge.
    edge_points: set[tuple[float, float]] = set()
    for e in boundary:
        pts = list(e.get_points("xy"))
        for x, y in pts:
            edge_points.add((round(x, 6), round(y, 6)))
    for original in synthetic_plot.boundary.points:
        assert any(abs(ex - original[0]) < 1e-5 and abs(ey - original[1]) < 1e-5
                   for ex, ey in edge_points)

    # Reconstruct polygon from edges and verify area is preserved.
    segs = [LineString(list(e.get_points("xy"))) for e in boundary]
    polys = list(polygonize(unary_union(segs)))
    assert polys, "boundary LWPOLYLINE entities must form a closed polygon"
    assert polys[0].area == pytest.approx(synthetic_plot.boundary.computed_area, abs=1e-6)


def test_roundtrip_preserves_entity_counts(synthetic_plot: Plot, tmp_path: Path) -> None:
    path = write_dxf(synthetic_plot, tmp_path / "counts.dxf")
    msp = ezdxf.readfile(path).modelspace()
    # EVERY corner is written (labelled or synthetic "?N") -- restores the
    # documented invariant (red-fill count == STONES count) and feeds M2 all
    # stone positions.
    assert len(by_layer(msp, LayerType.STONES)) == len(synthetic_plot.corner_points)
    dim_total = sum(
        len(by_layer(msp, layer))
        for layer in (LayerType.BOUNDARY_DIMENSIONS, LayerType.CHAINLINE_DIMENSIONS,
                      LayerType.DIMENSIONS)
    )
    assert dim_total == len(synthetic_plot.measurements)


# --- Real fixture end-to-end -------------------------------------------------


def test_real_plot_serializes_and_roundtrips(tmp_path: Path) -> None:
    f = FMB_DIR / "FMB_SIVAGANGAI_Manamadurai_T.Pudukkottai_100.pdf"
    vectors = extract_vectors(f)
    detections = extract_text(f)
    plot = build_plot(
        client_id="client_test",
        vectors=vectors,
        detections=detections,
        anchor_result=anchor_measurements(vectors, detections),
        header=parse_header(detections),
    )
    path = write_dxf(plot, tmp_path / "survey100.dxf")
    msp = ezdxf.readfile(path).modelspace()

    # Boundary: one LWPOLYLINE per edge, forms a closed polygon.
    boundary = by_layer(msp, LayerType.BOUNDARY)
    assert len(boundary) >= 1
    assert all(e.dxftype() == "LWPOLYLINE" for e in boundary)
    # Every corner stone is written (red-fill count == STONES count invariant).
    assert len(by_layer(msp, LayerType.STONES)) == len(plot.corner_points)
    assert [e.dxf.text for e in by_layer(msp, LayerType.SURVEY_NUMBER)] == ["100"]

    # Area survives the round-trip: reconstruct polygon from LWPOLYLINE edges.
    segs = [LineString(list(e.get_points("xy"))) for e in boundary]
    polys = list(polygonize(unary_union(segs)))
    assert polys, "boundary LWPOLYLINE entities must form a closed polygon"
    assert polys[0].area == pytest.approx(plot.boundary.computed_area, rel=1e-4)

    # Subdivision lines land on their layer (survey 100 has internal segments).
    assert len(by_layer(msp, LayerType.SUBDIVISION_LINES)) == len(plot.subdivision_segments)
    # Chain layer: survey 100 has 0 chain segments per fixture counts.
    assert len(by_layer(msp, LayerType.CHAIN_LINES)) == len(plot.chain_segments)


# --- Text height --------------------------------------------------------------


def test_text_height_scales_with_boundary(synthetic_plot: Plot) -> None:
    """Base text height is ~1.4% of the bounding box (manual-DXF calibration).

    Calibrated so survey 100 (extent ~215 m) yields dimension height ~3.0,
    matching the client's manual DXF. Clamped to [1.5, 6].
    """
    # synthetic_plot boundary: (0,0)-(10,10), max_dim = 10m.
    # 1.4% of 10 = 0.14, but floor is 1.5.
    h = text_height_for(synthetic_plot)
    assert h == pytest.approx(1.5)

    survey100_like = Plot(
        client_id="c", survey_no="x", district="", taluk="", village="",
        boundary=Boundary(points=[(0, 0), (215, 0), (215, 106), (0, 106), (0, 0)]),
    )
    # 1.4% of 215 = 3.01 → matches the manual's dimension height of 3.0.
    assert text_height_for(survey100_like) == pytest.approx(3.01, abs=0.05)

    very_large = Plot(
        client_id="c", survey_no="x", district="", taluk="", village="",
        boundary=Boundary(points=[(0, 0), (600, 0), (600, 100), (0, 100), (0, 0)]),
    )
    # 1.4% of 600 = 8.4 → clamped to 6.0.
    assert text_height_for(very_large) == pytest.approx(6.0)


def test_segments_written_to_correct_layers() -> None:
    """Subdivision and chain segments land on their layers as LINE entities."""
    plot = Plot(
        client_id="c", survey_no="1", district="", taluk="", village="",
        boundary=Boundary(points=[(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]),
        subdivision_segments=[((10, 0), (10, 100)), ((0, 50), (100, 50))],
        chain_segments=[((0, 0), (100, 100))],
    )
    msp = build_document(plot).modelspace()

    sub_lines = by_layer(msp, LayerType.SUBDIVISION_LINES)
    assert len(sub_lines) == 2
    assert all(e.dxftype() == "LWPOLYLINE" for e in sub_lines)

    chain_lines = by_layer(msp, LayerType.CHAIN_LINES)
    assert len(chain_lines) == 1
    assert chain_lines[0].dxftype() == "LWPOLYLINE"


# --- In-plot label alignment --------------------------------------------------


def test_inplot_labels_are_centred_on_their_anchor() -> None:
    """Sub-plot and neighbor labels are MIDDLE_CENTER (centred on their point).

    They used to default to LEFT/BASELINE (``halign=0/valign=0``), so the glyph
    started at the anchor and ran up-and-right instead of sitting centred on it —
    the in-plot misalignment.  Survey number and dimension labels were already
    centred; this locks every in-plot label to the same convention.  STONES stay
    LEFT (their grip must sit on the line-endpoint node) — asserted elsewhere.
    """
    plot = Plot(
        client_id="c", survey_no="42", district="", taluk="", village="",
        boundary=Boundary(points=[(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]),
        sub_plots=[
            SubPlot(label="2A", boundary=Boundary(
                points=[(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)])),
        ],
        sub_plot_labels=[SubPlotLabel(label="3B", position=(70.0, 70.0))],
        neighbor_labels=[NeighborLabel(label="99", position=(50.0, 5.0))],
    )
    msp = build_document(plot).modelspace()

    sub = by_layer(msp, LayerType.SUBDIVISION)
    assert sub, "expected SUBDIVISION labels"
    for e in sub:
        assert (e.dxf.halign, e.dxf.valign) == (1, 2)  # MIDDLE_CENTER
        # The centred insertion point coincides with the align point (the anchor).
        assert e.dxf.align_point is not None

    nbr = by_layer(msp, LayerType.NEIGHBOR_LABEL)
    assert nbr, "expected neighbor labels"
    for e in nbr:
        assert (e.dxf.halign, e.dxf.valign) == (1, 2)  # MIDDLE_CENTER

    # Sub-plot label '3B' is centred on its detection point (70, 70).
    spl = next(e for e in sub if e.dxf.text == "3B")
    ap = spl.dxf.align_point
    assert (round(ap.x, 3), round(ap.y, 3)) == (70.0, 70.0)


def _ring_centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    """Mirror to_dxf._centroid: plain average of the (closed) ring's points."""
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def test_survey_number_nudges_off_a_colliding_measurement() -> None:
    """The big survey number lifts off a measurement label sitting at the centroid.

    On dense plots an internal/chain dimension lands on the boundary centroid, so
    the large survey glyph printed on top of it.  ``build_document`` now nudges the
    survey number straight up to the nearest clear spot — same x, higher y — so it
    stays central but no longer overlaps the measurement.
    """
    ring = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    centroid = _ring_centroid(ring)
    plot = Plot(
        client_id="c", survey_no="698", district="", taluk="", village="",
        boundary=Boundary(points=ring),
        # A dimension label sitting right on the centroid.
        measurements=[Measurement(raw="154.4", line_class="internal", position=centroid)],
    )
    msp = build_document(plot).modelspace()

    survey = by_layer(msp, LayerType.SURVEY_NUMBER)[0]
    sx, sy = survey.dxf.align_point.x, survey.dxf.align_point.y
    assert round(sx, 3) == round(centroid[0], 3)  # stays horizontally central
    assert sy > centroid[1]                        # lifted up off the measurement

    dim = by_layer(msp, LayerType.DIMENSIONS)[0]
    dx, dy = dim.dxf.align_point.x, dim.dxf.align_point.y
    # The two label centres no longer coincide (they did before the nudge).
    assert (round(sx, 1), round(sy, 1)) != (round(dx, 1), round(dy, 1))


def test_survey_number_stays_at_centroid_when_no_collision() -> None:
    """With no measurement on the centroid, the survey number is left centred there."""
    ring = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
    centroid = _ring_centroid(ring)
    plot = Plot(
        client_id="c", survey_no="7", district="", taluk="", village="",
        boundary=Boundary(points=ring),
        measurements=[Measurement(raw="40.0", line_class="boundary", position=(50.0, 2.0))],
    )
    msp = build_document(plot).modelspace()
    survey = by_layer(msp, LayerType.SURVEY_NUMBER)[0]
    ap = survey.dxf.align_point
    assert (round(ap.x, 3), round(ap.y, 3)) == (round(centroid[0], 3), round(centroid[1], 3))
