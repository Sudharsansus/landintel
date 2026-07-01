"""Static connection test — verifies the full M1 data flow without running OCR or loading any model.

Tests every interface handoff:
  OCRDetection → anchor_measurements → build_plot → write_dxf

Uses synthetic fixtures that match the real data format exactly.
No PDF, no PaddleOCR, no GPU required.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from landintel.pipeline.m1_extract.anchor import (
    MAX_ANCHOR_DISTANCE,
    MAX_ANGLE_DIFF,
    AnchorResult,
    anchor_measurements,
)
from landintel.pipeline.m1_extract.build_plot import build_plot
from landintel.pipeline.m1_extract.ocr import FmbHeader, OCRDetection
from landintel.pipeline.m1_extract.pdf_vectors import Marker, PageVectors, Segment
from landintel.pipeline.m1_extract.to_dxf import write_dxf


# ── Synthetic fixtures ────────────────────────────────────────────────────────


def _seg(x1: float, y1: float, x2: float, y2: float, width: float = 2.0) -> Segment:
    return Segment(start=(x1, y1), end=(x2, y2), width=width)


def _det(
    text: str,
    cx: float,
    cy: float,
    angle_deg: float = 0.0,
    confidence: float = 0.95,
    kind: str = "decimal_measurement",
) -> OCRDetection:
    """Build an OCRDetection centred at (cx, cy) with the given angle."""
    r = math.radians(angle_deg)
    hw, hh = 10.0, 4.0
    ca, sa = math.cos(r), math.sin(r)
    tl = (cx - ca * hw + sa * hh, cy - sa * hw - ca * hh)
    tr = (cx + ca * hw + sa * hh, cy + sa * hw - ca * hh)
    br = (cx + ca * hw - sa * hh, cy + sa * hw + ca * hh)
    bl = (cx - ca * hw - sa * hh, cy - sa * hw + ca * hh)
    return OCRDetection(
        text=text,
        confidence=confidence,
        polygon=(tl, tr, br, bl),
        angle_deg=angle_deg,
        kind=kind,
    )


def _vectors() -> PageVectors:
    """A simple closed rectangular boundary + one internal segment."""
    return PageVectors(
        boundary=[
            _seg(100, 700, 300, 700),  # bottom edge, horizontal
            _seg(300, 700, 300, 200),  # right edge, vertical
            _seg(300, 200, 100, 200),  # top edge, horizontal
            _seg(100, 200, 100, 700),  # left edge, vertical
        ],
        internal=[_seg(100, 450, 300, 450, width=1.0)],
        chain=[_seg(100, 700, 300, 200, width=1.0)],
        separation=[_seg(95, 200, 95, 700, width=1.0)],
        stones=[
            Marker(x=100, y=700, width=5.0, height=5.0),
            Marker(x=300, y=700, width=5.0, height=5.0),
            Marker(x=300, y=200, width=5.0, height=5.0),
            Marker(x=100, y=200, width=5.0, height=5.0),
        ],
        page_width=595.0,
        page_height=841.0,
    )


def _header() -> FmbHeader:
    return FmbHeader(
        survey_no="405",
        district="Vellore",
        taluk="Gudiyatham",
        village="Kallapadi",
        scale_denominator=1079,
        stated_area_ha=1.44,
    )


def _detections() -> list[OCRDetection]:
    """Measurements placed near their lines + non-measurement labels."""
    return [
        # Near bottom edge (y=700, horizontal → angle=0)
        _det("44.2", cx=200, cy=715, angle_deg=0.0),
        # Near right edge (x=300, vertical → angle=90)
        _det("31.2", cx=315, cy=450, angle_deg=90.0),
        # Near top edge (y=200, horizontal → angle=0)
        _det("41.0", cx=200, cy=185, angle_deg=0.0),
        # Near left edge (x=100, vertical → angle=90)
        _det("74.0", cx=85, cy=450, angle_deg=90.0),
        # Corner labels — should go to unanchored (_NON_MEASUREMENT_RE)
        _det("A", cx=100, cy=700, angle_deg=0.0, kind="corner_label"),
        _det("B", cx=300, cy=700, angle_deg=0.0, kind="corner_label"),
        # Neighbour survey number — should go to unanchored
        _det("403", cx=50, cy=450, angle_deg=90.0, kind="integer_marker"),
        # Header text — far from any line, should go to unanchored
        _det(
            "District : Vellore",
            cx=300,
            cy=50,
            angle_deg=0.0,
            confidence=0.98,
            kind="unknown",
        ),
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestOCRDetectionContract:
    """OCRDetection fields and center property."""

    def test_center_is_centroid(self) -> None:
        d = _det("44.2", cx=200.0, cy=715.0, angle_deg=0.0)
        cx, cy = d.center
        assert abs(cx - 200.0) < 1.0
        assert abs(cy - 715.0) < 1.0

    def test_polygon_has_four_corners(self) -> None:
        d = _det("44.2", cx=200.0, cy=715.0)
        assert len(d.polygon) == 4

    def test_angle_deg_in_range(self) -> None:
        for angle in (0.0, 45.0, 90.0, 135.0, 179.9):
            d = _det("x", 100, 100, angle_deg=angle)
            assert d.angle_deg is not None
            assert 0.0 <= d.angle_deg < 180.0

    def test_fields_present(self) -> None:
        d = _det("44.2", 100, 100)
        assert isinstance(d.text, str)
        assert 0.0 <= d.confidence <= 1.0
        assert isinstance(d.polygon, tuple)
        assert d.kind in (
            "decimal_measurement",
            "integer_marker",
            "corner_label",
            "chain_point",
            "red_token",
            "unknown",
        )


class TestAnchorConnection:
    """OCRDetection → AnchorResult handoff."""

    @pytest.fixture(scope="class")
    def result(self) -> AnchorResult:
        return anchor_measurements(_vectors(), _detections())

    def test_returns_anchor_result(self, result: AnchorResult) -> None:
        assert isinstance(result, AnchorResult)

    def test_accounting_complete(self, result: AnchorResult) -> None:
        total = (
            len(result.anchored)
            + len(result.unanchored_measurements)
            + len(result.sub_plot_detections)
            + len(result.neighbor_labels)
            + len(result.chain_dim_detections)
        )
        assert total == len(_detections())

    def test_measurements_anchored(self, result: AnchorResult) -> None:
        anchored_texts = {m.text for m in result.anchored}
        for value in ("44.2", "31.2", "41.0", "74.0"):
            assert value in anchored_texts, f"{value} should be anchored"

    def test_non_measurements_unanchored(self, result: AnchorResult) -> None:
        unanchored_texts = {d.text for d in result.unanchored_measurements}
        assert "A" in unanchored_texts
        assert "B" in unanchored_texts
        # "403" is a 3-digit number → routed to neighbor_labels, not unanchored
        neighbor_texts = {d.text for d in result.neighbor_labels}
        assert "403" in neighbor_texts

    def test_anchored_within_gates(self, result: AnchorResult) -> None:
        for m in result.anchored:
            assert m.distance <= MAX_ANCHOR_DISTANCE
            assert m.angle_diff <= MAX_ANGLE_DIFF

    def test_angle_deg_preferred_over_polygon(self, result: AnchorResult) -> None:
        # angle_deg is set on all four measurement detections → all four anchor
        assert len(result.anchored) == 4

    def test_anchored_measurement_fields(self, result: AnchorResult) -> None:
        for m in result.anchored:
            assert isinstance(m.text, str)
            assert 0.0 <= m.confidence <= 1.0
            assert isinstance(m.center, tuple) and len(m.center) == 2
            assert m.line_class in ("boundary", "internal", "chain")
            assert hasattr(m.line, "start") and hasattr(m.line, "end")
            assert isinstance(m.distance, float)
            assert isinstance(m.angle_diff, float)


class TestBuildPlotConnection:
    """AnchorResult + PageVectors + FmbHeader → Plot handoff."""

    @pytest.fixture(scope="class")
    def plot(self):  # type: ignore[no-untyped-def]
        vectors = _vectors()
        detections = _detections()
        anchor_result = anchor_measurements(vectors, detections)
        return build_plot(
            client_id="test",
            vectors=vectors,
            detections=detections,
            anchor_result=anchor_result,
            header=_header(),
        )

    def test_plot_fields_present(self, plot) -> None:  # type: ignore[no-untyped-def]
        assert plot.survey_no == "405"
        assert plot.district == "Vellore"
        assert plot.scale == 1079
        assert plot.stated_area == pytest.approx(1.44)

    def test_boundary_in_metres(self, plot) -> None:  # type: ignore[no-untyped-def]
        from landintel.pipeline.m1_extract.build_plot import points_to_metres

        assert plot.boundary is not None
        pts = plot.boundary.points
        assert len(pts) >= 4
        xs = [p[0] for p in pts]
        span = max(xs) - min(xs)
        # scale=1079; ppm = (2.54/72)*1079/100 ≈ 0.381 m/pt; 200px → ≈76.1m
        expected = 200.0 * points_to_metres(1079)
        assert abs(span - expected) < 1.0, f"boundary span {span:.2f}m, expected ≈{expected:.2f}m"

    def test_measurements_have_position(self, plot) -> None:  # type: ignore[no-untyped-def]
        assert len(plot.measurements) > 0
        for m in plot.measurements:
            assert m.position is not None
            px, py = m.position
            assert not (abs(px) < 0.01 and abs(py) < 0.01), (
                f"measurement '{m.raw}' has zero position — geometric placement failed"
            )

    def test_measurements_raw_text(self, plot) -> None:  # type: ignore[no-untyped-def]
        raws = {m.raw for m in plot.measurements}
        for value in ("44.2", "31.2", "41.0", "74.0"):
            assert value in raws

    def test_subdivision_segments_present(self, plot) -> None:  # type: ignore[no-untyped-def]
        assert len(plot.subdivision_segments) >= 1

    def test_chain_segments_present(self, plot) -> None:  # type: ignore[no-untyped-def]
        assert len(plot.chain_segments) >= 1

    def test_separation_segments_present(self, plot) -> None:  # type: ignore[no-untyped-def]
        assert len(plot.separation_segments) >= 1

    def test_corner_points_present(self, plot) -> None:  # type: ignore[no-untyped-def]
        assert len(plot.corner_points) == 4


class TestDxfWriterConnection:
    """Plot → DXF file handoff."""

    @pytest.fixture(scope="class")
    def dxf_path(self, tmp_path_factory):  # type: ignore[no-untyped-def]
        vectors = _vectors()
        detections = _detections()
        anchor_result = anchor_measurements(vectors, detections)
        plot = build_plot(
            client_id="test",
            vectors=vectors,
            detections=detections,
            anchor_result=anchor_result,
            header=_header(),
        )
        out = tmp_path_factory.mktemp("dxf") / "survey_405_test.dxf"
        return write_dxf(plot, out)

    def test_dxf_file_created(self, dxf_path: Path) -> None:
        assert dxf_path.exists()
        assert dxf_path.stat().st_size > 1000

    def test_dxf_opens_with_ezdxf(self, dxf_path: Path) -> None:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        assert doc is not None

    def test_all_layers_present(self, dxf_path: Path) -> None:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        layer_names = {layer.dxf.name for layer in doc.layers}
        expected = {
            "BOUNDARY",
            "SUBDIVISION",
            "CHAIN_LINES",
            "SEPARATION_LINE",
            "BOUNDARY_DIMENSIONS",
            "CHAINLINE_DIMENSIONS",
            "SURVEY_NUMBER",
            "SUBDIVISION_LINES",
            "WELL and BUILDING",
            "STONES",
            "DIMENSIONS",
        }
        for name in expected:
            assert name in layer_names, f"Layer '{name}' missing from DXF"

    def test_boundary_written_as_separate_lines(self, dxf_path: Path) -> None:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        bnd_poly = [
            e for e in msp
            if e.dxf.layer == "BOUNDARY" and e.dxftype() == "LWPOLYLINE"
        ]
        assert len(bnd_poly) >= 4
        bnd_line = [
            e for e in msp
            if e.dxf.layer == "BOUNDARY" and e.dxftype() == "LINE"
        ]
        assert len(bnd_line) == 0

    def test_measurement_texts_in_dxf(self, dxf_path: Path) -> None:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        dim_layers = {"BOUNDARY_DIMENSIONS", "CHAINLINE_DIMENSIONS", "DIMENSIONS"}
        texts = {
            e.dxf.text
            for e in msp
            if e.dxftype() == "TEXT" and e.dxf.layer in dim_layers
        }
        assert any("," in t or "." in t for t in texts), (
            f"No measurement texts found, got: {texts}"
        )

    def test_survey_number_in_dxf(self, dxf_path: Path) -> None:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        survey_texts = [
            e.dxf.text
            for e in msp
            if e.dxftype() == "TEXT" and e.dxf.layer == "SURVEY_NUMBER"
        ]
        assert "405" in survey_texts

    def test_layer_colors_correct(self, dxf_path: Path) -> None:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        color_spec = {
            "BOUNDARY": 1,
            "SUBDIVISION_LINES": 3,
            "BOUNDARY_DIMENSIONS": 220,
            "SURVEY_NUMBER": 1,
            "SUBDIVISION": 130,
            "WELL and BUILDING": 5,
            "STONES": 7,
            "DIMENSIONS": 200,
        }
        for layer_name, expected_color in color_spec.items():
            layer = doc.layers.get(layer_name)
            assert layer is not None, f"Layer '{layer_name}' not found"
            assert layer.color == expected_color, (
                f"Layer '{layer_name}': expected color {expected_color}, got {layer.color}"
            )

    def test_linetypes_registered(self, dxf_path: Path) -> None:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        lt_names = {lt.dxf.name for lt in doc.linetypes}
        assert "CHAINLINE" in lt_names
        assert "DASHDOT" in lt_names
