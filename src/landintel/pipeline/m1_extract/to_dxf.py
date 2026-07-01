"""Serialize an assembled Plot into a layered DXF.

Single responsibility: pure serialization. Every geometry decision was already
made in ``build_plot`` (scale applied, ring chained, area computed); this module
only writes what the :class:`~landintel.core.models.Plot` already holds onto the
right layers.

Layer fidelity is the bar. The output uses the exact layer names from
:class:`~landintel.core.enums.LayerType` (grounded against the client's own
DXFs) and every canonical layer is created -- even when empty -- so the file
structurally matches the client's convention and drops into their AutoCAD
workflow unchanged. Each element is written to its correct layer:

* boundary ring     -> ``BOUNDARY``              one LWPOLYLINE per edge, red (ACI 1)
* separation ticks  -> ``SEPARATION_LINE``        DASHDOT
* subdivision lines -> ``SUBDIVISION_LINES``      LWPOLYLINE, green (ACI 3)
* chain lines       -> ``CHAIN_LINES``            CHAINLINE linetype, dark gray (ACI 251)
* corner stones     -> ``STONES``                 TEXT only (no POINT), white/black (ACI 7)
* survey number     -> ``SURVEY_NUMBER``          TEXT, red (ACI 1) — one entity
* sub-plot labels   -> ``SUBDIVISION``            TEXT, teal (ACI 130)
* measurements      -> ``BOUNDARY_DIMENSIONS`` / ``CHAINLINE_DIMENSIONS`` /
                       ``DIMENSIONS`` by line class; dot->comma (44.2->44,2)

Coordinates are real-world metres -- what ``build_plot`` produced.
"""

from __future__ import annotations

import logging
from pathlib import Path

import ezdxf

_log = logging.getLogger(__name__)
from ezdxf.document import Drawing
from ezdxf.enums import TextEntityAlignment

from ...core.enums import LayerType
from ...core.models import Boundary, Plot, Point

__all__ = ["build_document", "write_dxf", "text_height_for"]

_DXF_VERSION = "R2010"

# Which dimension layer a measurement goes to, by the line class it labels.
_DIMENSION_LAYER = {
    "boundary": LayerType.BOUNDARY_DIMENSIONS,
    "chain":    LayerType.CHAINLINE_DIMENSIONS,
    "internal": LayerType.DIMENSIONS,
}

# ACI color numbers by layer.  Verified against client DXF fixtures.
_LAYER_COLOR: dict[LayerType, int] = {
    LayerType.BOUNDARY:             1,    # Red
    LayerType.SUBDIVISION_LINES:    3,    # Green
    LayerType.BOUNDARY_DIMENSIONS:  220,
    LayerType.CHAINLINE_DIMENSIONS: 251,  # Dark gray — matches CHAIN_LINES
    LayerType.CHAIN_LINES:          251,  # Dark gray
    LayerType.DIMENSIONS:           200,  # Sub-division dimension labels
    LayerType.STONES:               7,    # White/Black
    LayerType.SURVEY_NUMBER:        1,    # Red
    LayerType.SUBDIVISION:          130,  # Teal
    LayerType.WELL_AND_BUILDING:    5,    # Blue
    LayerType.SEPARATION_LINE:      7,    # White/Black
    LayerType.DASHED_REF:           7,    # White/Black
    LayerType.NEIGHBOR_LABEL:       7,    # White
}

# Non-continuous linetypes by layer.
_LAYER_LINETYPE: dict[LayerType, str] = {
    LayerType.CHAIN_LINES:     "CHAINLINE",
    LayerType.SEPARATION_LINE: "DASHDOT",
    LayerType.DASHED_REF:      "DASHED",
}


def _load_linetypes(doc: Drawing) -> None:
    """Register CHAINLINE, DASHDOT, DOT, and DASHED linetypes if not already present."""
    if "CHAINLINE" not in doc.linetypes:
        lt = doc.linetypes.new("CHAINLINE")
        lt.dxf.description = "Chainline ____ . . . ____"
        # Match the client's manual DXF pattern exactly: a 30-unit dash, then
        # three short ticks, then a 30-unit dash.  Our coordinates are in metres
        # (extent ~215 m), the same scale as the manual, so the same absolute
        # pattern renders as the visible dash-dot the client expects.  The old
        # 1.25-unit pattern repeated ~100x across the plot and looked solid.
        lt.setup_pattern([30.0, -10.0, 1.0, -3.0, 1.0, -3.0, 1.0, -10.0])
    if "DOT" not in doc.linetypes:
        lt = doc.linetypes.new("DOT")
        lt.dxf.description = "Dotted line . . . . . . . . . ."
        lt.setup_pattern([0.0, -0.25])
    if "DASHDOT" not in doc.linetypes:
        lt = doc.linetypes.new("DASHDOT")
        lt.dxf.description = "Dash dot __ . __ . __ . __ . __"
        lt.setup_pattern([0.5, -0.25, 0.0, -0.25])
    if "DASHED" not in doc.linetypes:
        lt = doc.linetypes.new("DASHED")
        lt.dxf.description = "Dashed __ __ __ __ __ __ __"
        lt.setup_pattern([0.5, -0.25])


def text_height_for(plot: Plot) -> float:
    """Base DXF text height (metres) for dimension labels, scaled to the plot.

    Calibrated to the client's manual DXF: on survey 100 (boundary extent
    ~215 m) the manual uses height 3.0 for dimension text, i.e. ~1.4% of the
    larger extent — not 3%.  The old 3% made every label 2-3x oversized, which
    cluttered the map and read as "wrong" against the clean manual reference.
    Per-layer multipliers in ``build_document`` derive stone/survey/sub-plot
    heights from this base (manual ratios: stones 2.0, sub-plot 4.0, survey 5.0).
    Clamped to [1.5, 6] m so tiny or huge plots stay legible.
    """
    if not (plot.boundary and plot.boundary.points):
        return 3.0
    pts = plot.boundary.points
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    max_dim = max(max(xs) - min(xs), max(ys) - min(ys))
    return max(1.5, min(6.0, max_dim * 0.014))


# Text style used by the client's manual DXF for sub-plot labels and the
# survey number (everything else is STANDARD).
_TIMES_STYLE = "TIMES_NEW_ROMAN"


def _ensure_text_styles(doc: Drawing) -> None:
    """Register the TIMES_NEW_ROMAN text style the manual DXF uses."""
    if _TIMES_STYLE not in doc.styles:
        doc.styles.new(_TIMES_STYLE, dxfattribs={"font": "times.ttf"})


def _ensure_layers(doc: Drawing) -> None:
    """Create every canonical layer with the correct color and linetype."""
    _load_linetypes(doc)
    _ensure_text_styles(doc)
    for layer_type in LayerType:
        name = layer_type.value
        if name in doc.layers:
            layer = doc.layers.get(name)
        else:
            layer = doc.layers.add(name)
        if layer_type in _LAYER_COLOR:
            layer.color = _LAYER_COLOR[layer_type]
        if layer_type in _LAYER_LINETYPE:
            layer.dxf.linetype = _LAYER_LINETYPE[layer_type]


def _centroid(boundary: Boundary) -> Point:
    """Average of the ring's points (placement anchor for the survey number)."""
    pts = boundary.points
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def _dot_to_comma(text: str) -> str:
    """Replace decimal dot with comma (44.2 -> 44,2) as per client convention."""
    return text.replace(".", ",")


def _text_half_extents(text: str, height: float) -> tuple[float, float]:
    """Half-width / half-height of a centred TEXT label (rough, axis-aligned).

    Width estimate uses the same ~0.6·height-per-glyph ratio ``build_plot`` uses
    for its overlap boxes, so the two stages agree on label footprints.
    """
    return (len(text) * height * 0.6) / 2.0 + 0.5, height / 2.0 + 0.5


def _clear_survey_position(
    centroid: Point,
    survey_no: str,
    survey_height: float,
    measurements: list,
    base_height: float,
    *,
    max_steps: int = 8,
) -> Point:
    """Find a placement for the big survey number that clears measurement labels.

    The survey number is drawn centred on the boundary centroid, but a measurement
    label (an internal/chain dimension) can land on the same spot, so the large
    glyph prints on top of a number (e.g. survey 698: "698" over "154,4").  This
    nudges the survey number straight up in small steps until its centred box no
    longer overlaps any measurement label box, keeping it central while lifting it
    clear.  Returns the centroid unchanged when nothing collides.
    """
    s_hw, s_hh = _text_half_extents(survey_no, survey_height)
    boxes: list[tuple[float, float, float, float]] = []
    for m in measurements:
        if m.position is None:
            continue
        hw, hh = _text_half_extents(_dot_to_comma(m.raw), base_height)
        mx, my = m.position
        boxes.append((mx - hw, my - hh, mx + hw, my + hh))
    if not boxes:
        return centroid

    def overlaps(cx: float, cy: float) -> bool:
        a = (cx - s_hw, cy - s_hh, cx + s_hw, cy + s_hh)
        for b in boxes:
            if not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]):
                return True
        return False

    cx, cy = centroid
    if not overlaps(cx, cy):
        return centroid
    # Try stepping the label up then down by one label-height at a time and take
    # the nearest clear spot (up preferred — keeps it reading above its cell).
    step = survey_height * 1.2
    for k in range(1, max_steps + 1):
        for sign in (1.0, -1.0):
            cand = (cx, cy + sign * step * k)
            if not overlaps(*cand):
                return cand
    return centroid


def build_document(plot: Plot) -> Drawing:
    """Build an in-memory DXF document from ``plot`` (no file I/O).

    Split out from :func:`write_dxf` so tests can round-trip without touching the
    filesystem.
    """
    doc = ezdxf.new(_DXF_VERSION)
    _ensure_layers(doc)
    msp = doc.modelspace()
    # Base height = dimension text height. Per-layer heights follow the client's
    # manual DXF ratios (dims 3.0, stones 2.0, sub-plot 4.0, survey 5.0 on the
    # survey-100 extent → base, base·⅔, base·4⁄3, base·5⁄3).
    height = text_height_for(plot)
    stone_height = height * (2.0 / 3.0)
    subplot_height = height * (4.0 / 3.0)
    survey_height = height * (5.0 / 3.0)

    def _add_seg(start: Point, end: Point, layer: str) -> None:
        """Add a 2-point LWPOLYLINE (matches the client DXF entity type)."""
        msp.add_lwpolyline([start, end], dxfattribs={"layer": layer})

    # Dashed neighbor-boundary reference ticks
    for start, end in plot.dashed_ref_segments:
        _add_seg(start, end, LayerType.DASHED_REF.value)

    # Subdivision geometry (green internal lines)
    for start, end in plot.subdivision_segments:
        _add_seg(start, end, LayerType.SUBDIVISION_LINES.value)

    # Chain/traverse geometry (CHAINLINE linetype — set explicitly on entities
    # to match the client reference which does not rely on BYLAYER inheritance)
    for start, end in plot.chain_segments:
        msp.add_lwpolyline(
            [start, end],
            dxfattribs={"layer": LayerType.CHAIN_LINES.value, "linetype": "CHAINLINE"},
        )

    # Separation ticks (DASHDOT)
    for start, end in plot.separation_segments:
        _add_seg(start, end, LayerType.SEPARATION_LINE.value)

    # Boundary — one LWPOLYLINE per edge, red.
    if plot.boundary is not None and plot.boundary.points:
        pts = plot.boundary.points
        for i in range(len(pts) - 1):
            _add_seg(pts[i], pts[i + 1], LayerType.BOUNDARY.value)

    # Corner stones — TEXT only (manual reference has TEXT entities only here).
    # Justify = LEFT (halign=Left, valign=Baseline): the TEXT insertion point
    # (group 10 — the single grip you click in AutoCAD) is anchored EXACTLY on
    # the stone's connecting point (``x, y`` = the line junction).  So the stone
    # number's grip already coincides with the line-endpoint node (no manual
    # "join to node" step), and the number reads up-and-right of the corner.
    # Write EVERY corner stone, including position-only ones whose OCR label
    # came back empty. Stone POSITIONS are load-bearing for M2 (they are the
    # georef anchors); the label is best-effort. Dropping unlabeled stones broke
    # the documented invariant (PDF red-fill count == DXF STONES count) and
    # starved M2 of corners on low-OCR-recall districts (INGUR: 43 stones -> 14
    # written). A position-only stone gets a synthetic ``?N`` placeholder so the
    # TEXT entity exists for M2 to read; M2 matches on geometry, never the label.
    for idx, corner in enumerate(plot.corner_points):
        label = corner.label if corner.label else f"?{idx + 1}"
        text = msp.add_text(
            label,
            height=stone_height,
            dxfattribs={"layer": LayerType.STONES.value},
        )
        text.set_placement(
            (corner.x, corner.y), align=TextEntityAlignment.LEFT
        )

    # Sub-plot labels from full SubPlot objects (boundary-resolved, optional).
    # MIDDLE_CENTER so the number sits CENTRED on its sub-cell centroid rather
    # than starting there and running up-and-right (the LEFT/BASELINE default,
    # which made every sub-plot number look offset from its cell).
    for sub in plot.sub_plots:
        if sub.boundary is not None and sub.boundary.points:
            label = msp.add_text(
                sub.label,
                height=subplot_height,
                dxfattribs={"layer": LayerType.SUBDIVISION.value, "style": _TIMES_STYLE},
            )
            label.set_placement(
                _centroid(sub.boundary), align=TextEntityAlignment.MIDDLE_CENTER
            )

    # Sub-plot labels from OCR detections (2A, 3B, 5B, …) — placed CENTRED on the
    # detection position (where the surveyor drew the number, i.e. inside its
    # sub-cell); height follows the manual's SUBDIVISION ratio.  MIDDLE_CENTER
    # keeps the glyph centred on that point instead of LEFT-justified off it.
    # TIMES_NEW_ROMAN matches the client's manual DXF for this layer.
    for spl in plot.sub_plot_labels:
        lbl = msp.add_text(
            spl.label,
            height=subplot_height,
            dxfattribs={"layer": LayerType.SUBDIVISION.value, "style": _TIMES_STYLE},
        )
        lbl.set_placement(spl.position, align=TextEntityAlignment.MIDDLE_CENTER)

    # Measurement labels; dot->comma for all dimension layers (client convention)
    for measurement in plot.measurements:
        layer = _DIMENSION_LAYER.get(
            measurement.line_class or "", LayerType.DIMENSIONS
        )
        raw = _dot_to_comma(measurement.raw)
        rotation = measurement.line_angle if measurement.line_angle is not None else 0.0
        text = msp.add_text(
            raw,
            height=height,
            dxfattribs={"layer": layer.value, "rotation": rotation},
        )
        text.set_placement(
            measurement.position or _origin(plot),
            align=TextEntityAlignment.MIDDLE_CENTER,
        )

    # Neighbor survey-number labels near shared boundaries (Bug 3).
    # MIDDLE_CENTER so the number is centred on its detection point like every
    # other label, not LEFT-justified off it.
    for nb in plot.neighbor_labels:
        lbl = msp.add_text(
            nb.label,
            height=height,
            dxfattribs={"layer": LayerType.NEIGHBOR_LABEL.value},
        )
        lbl.set_placement(nb.position, align=TextEntityAlignment.MIDDLE_CENTER)

    # Survey number — one large TEXT entity at the boundary centroid.
    # The glyph's drawn position varies across PDFs (sometimes near a corner,
    # sometimes near the header edge); the centroid gives a stable, visually
    # central placement that matches the client reference convention.  When a
    # measurement label sits at that centroid the big glyph would print on top of
    # it, so nudge straight up/down to the nearest clear spot (stays central).
    if plot.survey_no and plot.boundary is not None and plot.boundary.points:
        position = _clear_survey_position(
            _centroid(plot.boundary),
            plot.survey_no,
            survey_height,
            plot.measurements,
            height,
        )
        text = msp.add_text(
            plot.survey_no,
            height=survey_height,
            dxfattribs={"layer": LayerType.SURVEY_NUMBER.value, "style": _TIMES_STYLE},
        )
        text.set_placement(position, align=TextEntityAlignment.MIDDLE_CENTER)

    return doc


def _origin(plot: Plot) -> Point:
    """Fallback placement when a measurement has no position."""
    if plot.boundary is not None and plot.boundary.points:
        return _centroid(plot.boundary)
    return (0.0, 0.0)


def write_dxf(plot: Plot, path: Path | str) -> Path:
    """Write ``plot`` to a DXF file at ``path`` and return the path."""
    out = Path(path)
    doc = build_document(plot)
    doc.saveas(out)
    n_bnd = len(plot.boundary.points) - 1 if (plot.boundary and plot.boundary.points) else 0
    _log.info(
        "DXF written [survey=%s]: %s bnd_edges=%d sep=%d measurements=%d stones=%d",
        plot.survey_no, out.name, n_bnd,
        len(plot.separation_segments),
        len(plot.measurements), len(plot.corner_points),
    )
    return out
