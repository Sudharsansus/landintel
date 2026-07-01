"""Extract and classify the vector geometry of an FMB PDF page.

Single responsibility: open a page, read its vector drawing paths, and sort them
into the feature classes the rest of M1 cares about -- boundary lines, separation
ticks, internal (subdivision) lines, chain lines, blue chain-markers and corner
stones -- by colour and stroke width.

Uses PATH-level two-pass classification (not item-level):
  Pass 1  — collect boundary endpoints from thick-black long paths
  Pass 2  — classify every path, using the endpoint set to detect:
              * short thick paths where both endpoints are already on a boundary
                path  ->  boundary connector (still boundary)
              * short thick paths where an endpoint is NOT on a boundary
                path  ->  separation tick  (S.F.SEPERATION LINE)
              * thin black paths where ALL segment endpoints match boundary
                points  ->  duplicate overlay (skip)

Classification rules (tunable constants, decoded empirically from the real
Sivagangai fixtures -- see notes below):

* Strokes are coloured black/grey, blue, or red. Black/grey strokes split into
  *boundary* (width >= BOUNDARY_MIN_WIDTH, long or connected) vs
  *separation* (width >= BOUNDARY_MIN_WIDTH, short, not connected) vs
  *internal* (thinner, non-duplicate).
  Blue strokes are *chain* lines.
* Fills are markers, classified purely by colour: blue fills are chain markers
  (arrows/dots), red fills are corner stones.

Empirical note on width: across all 46 Sivagangai fixtures boundary lines are
width 2.0 (occasionally 3.0) and internal lines are 1.0. The threshold is set
at 1.5 to separate the two and lives in a named constant so the next district --
which may differ again -- is a one-line change.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from ...core.models import Point

_log = logging.getLogger(__name__)

__all__ = [
    "Segment",
    "Marker",
    "PageVectors",
    "extract_vectors",
    "BOUNDARY_MIN_WIDTH",
    "_classify_glyph_groups",
]

# --- Classification constants (tunable) --------------------------------------

BOUNDARY_MIN_WIDTH: float = 1.5
"""Stroke width (pt) at/above which a black line is a boundary, below is internal."""

FRAME_PAGE_SPAN_RATIO: float = 0.85
"""A stroke spanning this fraction of a page axis is the page frame, not geometry.

The FMB sheet has a rectangular border/frame (and full-width separator lines)
drawn in black -- which the colour/width rules would otherwise file as
boundary/internal. The frame is not plot geometry: left in, it pollutes counts,
would be georeferenced as an edge in M2/M3, and sits beside the header so
anchoring mis-attaches labels to it.

The test is **axis-aware**: a stroke is frame if its horizontal extent reaches
this fraction of the page width *or* its vertical extent reaches this fraction
of the page height. A plain length threshold was wrong -- an elongated plot's
real long edge (e.g. ~480 pt on a narrow 57x370 plot) can exceed a frame-length
cutoff and be deleted, breaking closure. The frame spans ~0.9+ of a page axis
while real edges stay well under, so 0.85 separates them with margin.
"""

LONG_PATH_THRESHOLD_PT: float = 15.0
"""Fallback minimum total path length (PDF points) when scale is unknown.

When the scale is known (preferred) the threshold is computed from the
scale-aware constant ``LONG_PATH_THRESHOLD_M`` below, which is 10 m in all
districts. Use the fallback only when the PDF's scale header cannot be read.

At 1:2021 (Sivagangai), 15 pt ≈ 10.7 m — close enough.
At 1:1079 (Kallapadi), 15 pt ≈ 5.7 m — too short; use scale-aware threshold.
"""

LONG_PATH_THRESHOLD_M: float = 10.0
"""Real-world length threshold (metres) separating main boundary edges from
short separation ticks. Paths shorter than this that are not boundary-connected
go to the separation layer. Calibrated against reference fmb_ml extractor.
"""

_COLOR_DOMINANCE: float = 0.4
"""How far one RGB channel must exceed the others to count as that colour."""


# --- Returned data structures ------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """A single straight line segment in PDF page space, with its stroke width."""

    start: Point
    end: Point
    width: float


@dataclass(frozen=True)
class Marker:
    """A filled glyph (stone or chain arrow), reduced to its centre and size."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class PageVectors:
    """All classified vector primitives extracted from one FMB page."""

    boundary: list[Segment] = field(default_factory=list)
    internal: list[Segment] = field(default_factory=list)
    separation: list[Segment] = field(default_factory=list)
    chain: list[Segment] = field(default_factory=list)
    blue_markers: list[Marker] = field(default_factory=list)
    stones: list[Marker] = field(default_factory=list)
    dashed_ref: list[Segment] = field(default_factory=list)
    """Gray dashed neighbor-boundary reference ticks.

    Drawn in ~(0.25, 0.25, 0.25) gray with a dash pattern in some FMBs to mark
    shared boundaries with neighboring surveys.  Written to the ``DASHED_REF``
    (DASHED linetype) layer in DXF.
    """
    glyph_groups: list = field(default_factory=list)
    """Pre-classified text glyph clusters (list[GlyphGroup] from pdf_glyphs).

    Classification is done during ``extract_vectors()`` so that ``ocr.py`` can
    use exact vector positions rather than rasterized-page OCR positions. Each
    group's ``kind`` field encodes the semantic role:

    * ``"survey_number"`` — the largest blue cluster (the big plot-number glyph)
    * ``"parcel_label"``  — large blue cluster (1, 2A, 3B, 5B, ...)
    * ``"dimension"``     — small blue cluster (boundary/chain distances)
    * ``"red_label"``     — all red clusters (stone letters + neighbor numbers;
                            OCR value tells them apart downstream)
    * ``"neighbor"``      — black cluster near the page edge (adjacent survey no.)
    * ``"black_label"``   — other black cluster (boundary dimension in some PDFs)

    Groups are ordered: survey_number first, then parcel_label, then dimension,
    then red_label, then neighbor/black_label — so callers can iterate in
    semantic priority order.
    """
    page_width: float = 0.0
    page_height: float = 0.0

    def counts(self) -> dict[str, int]:
        """Number of primitives per class -- the handle tests assert against."""
        return {
            "boundary":   len(self.boundary),
            "internal":   len(self.internal),
            "separation": len(self.separation),
            "chain":      len(self.chain),
            "blue_markers": len(self.blue_markers),
            "stones":     len(self.stones),
            "dashed_ref": len(self.dashed_ref),
            "glyph_groups": len(self.glyph_groups),
        }


# --- Colour classification ---------------------------------------------------


def _color_name(color: tuple[float, float, float] | None) -> str:
    """Return ``"blue"``, ``"red"`` or ``"black"`` for an RGB tuple.

    Uses channel dominance with a tolerance so greys and near-blacks fall
    through to ``"black"`` and slightly-off primaries still match. ``None``
    (no colour set) is treated as black.
    """
    if color is None:
        return "black"
    r, g, b = color
    if b - max(r, g) > _COLOR_DOMINANCE:
        return "blue"
    if r - max(g, b) > _COLOR_DOMINANCE:
        return "red"
    return "black"


def _is_gray(color: tuple[float, float, float] | None) -> bool:
    """True for the gray page-border color (~0.25, 0.25, 0.25) used in some FMBs."""
    if color is None:
        return False
    return all(abs(v - 0.25) < 0.05 for v in color)


# --- Extraction --------------------------------------------------------------


def _scale_from_page(page: fitz.Page) -> int | None:
    """Extract FMB scale denominator from native page text (no OCR model).

    The FMB header text is rendered as native PDF text (unlike the body numbers
    which are vector glyph paths).  PyMuPDF reads it instantly without a model.
    Returns None when the scale line is absent or unreadable.
    """
    text = page.get_text("text")
    m = re.search(r"Scale\s*:\s*1\s*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _long_path_threshold_pt(scale_denominator: int | None) -> float:
    """Convert the 10 m threshold to PDF points for the given scale.

    Falls back to ``LONG_PATH_THRESHOLD_PT`` when scale is unknown.
    """
    if scale_denominator is None:
        return LONG_PATH_THRESHOLD_PT
    # points_to_metres: ppm = (2.54/72) * scale / 100
    ppm = (2.54 / 72.0) * scale_denominator / 100.0
    return LONG_PATH_THRESHOLD_M / ppm


def _line_segments(path: dict) -> list[tuple[Point, Point]]:
    """Return (start, end) for every straight-line item in a drawing path."""
    segs: list[tuple[Point, Point]] = []
    for item in path["items"]:
        if item[0] == "l":
            p1, p2 = item[1], item[2]
            segs.append(((p1.x, p1.y), (p2.x, p2.y)))
    return segs


def _to_marker(path: dict) -> Marker:
    """Reduce a filled path to a centre-point marker via its bounding box."""
    rect = path["rect"]
    return Marker(
        x=(rect.x0 + rect.x1) / 2.0,
        y=(rect.y0 + rect.y1) / 2.0,
        width=rect.width,
        height=rect.height,
    )


def extract_vectors(
    pdf_path: Path | str,
    *,
    page_number: int = 0,
    scale_denominator: int | None = None,
) -> PageVectors:
    """Extract and classify the vector primitives of one FMB PDF page.

    Uses a two-pass PATH-level classification:

    * Pass 1 collects all endpoints of thick-black paths whose total length
      exceeds the scale-aware threshold (10 m) -- these are the main boundary
      edges.  When ``scale_denominator`` is not provided the scale is read from
      the PDF's native header text; the fallback constant is used only when
      the scale cannot be determined at all.
    * Pass 2 classifies each path using those endpoints to distinguish boundary
      connectors from separation ticks (thick), and real subdivision lines from
      boundary duplicates (thin).

    Args:
        pdf_path: Path to the FMB PDF.
        page_number: Zero-based page index (FMB sheets are single-page).
        scale_denominator: FMB drawing-scale denominator (e.g. 2021 for
            "1 : 2021").  When omitted the function reads it from the PDF's
            native header text.  Pass explicitly when you already have it from
            ``parse_header`` to avoid re-reading the page.

    Returns:
        A :class:`PageVectors` with primitives sorted by feature class.
    """
    path = Path(pdf_path)
    with fitz.open(path) as doc:
        page = doc[page_number]
        pw = page.rect.width
        ph = page.rect.height
        frame_dx = FRAME_PAGE_SPAN_RATIO * pw
        frame_dy = FRAME_PAGE_SPAN_RATIO * ph

        scale = scale_denominator or _scale_from_page(page)
        threshold_pt = _long_path_threshold_pt(scale)
        _log.debug("vectors: scale=1:%s threshold_pt=%.1f", scale, threshold_pt)

        def _frame_filter(segs: list[tuple[Point, Point]]) -> list[tuple[Point, Point]]:
            return [
                (s, e) for s, e in segs
                if abs(e[0] - s[0]) < frame_dx and abs(e[1] - s[1]) < frame_dy
            ]

        # --- Pass 1: collect boundary endpoint pairs from long thick-black paths ---
        bnd_pts: set[tuple[int, int]] = set()
        bnd_pairs: set[frozenset[tuple[int, int]]] = set()
        for drawing in page.get_drawings():
            if drawing["type"] not in ("s", "fs"):
                continue
            color = drawing.get("color")
            if _color_name(color) != "black" or _is_gray(color):
                continue
            width = float(drawing.get("width") or 0.0)
            if width < BOUNDARY_MIN_WIDTH:
                continue
            segs = _frame_filter(_line_segments(drawing))
            if not segs:
                continue
            total = sum(math.hypot(e[0] - s[0], e[1] - s[1]) for s, e in segs)
            if total >= threshold_pt:
                for s, e in segs:
                    rs = (round(s[0]), round(s[1]))
                    re = (round(e[0]), round(e[1]))
                    bnd_pts.add(rs)
                    bnd_pts.add(re)
                    bnd_pairs.add(frozenset((rs, re)))

        # --- Pass 2: classify every path using the boundary endpoint set -----
        boundary: list[Segment] = []
        separation: list[Segment] = []
        internal: list[Segment] = []
        chain: list[Segment] = []
        blue_markers: list[Marker] = []
        stones: list[Marker] = []
        dashed_ref: list[Segment] = []

        for drawing in page.get_drawings():
            dtype = drawing["type"]
            has_stroke = dtype in ("s", "fs")
            has_fill = dtype in ("f", "fs")

            if has_stroke:
                color = drawing.get("color")
                if _is_gray(color):
                    # Gray dashed → neighbor-boundary reference tick; gray solid → frame, skip
                    if drawing.get("dashes"):
                        segs = _frame_filter(_line_segments(drawing))
                        w = float(drawing.get("width") or 0.0)
                        for s, e in segs:
                            dashed_ref.append(Segment(start=s, end=e, width=w))
                    continue
                stroke_color = _color_name(color)
                width = float(drawing.get("width") or 0.0)
                segs = _frame_filter(_line_segments(drawing))
                if not segs:
                    pass  # no stroke segments; may still have fill
                elif drawing.get("dashes", "").startswith("[ 30 10"):
                    # FMB chain / traverse lines: black dashed with the long-dash
                    # dot-dot pattern [ 30 10 1 3 1 3 1 10 ].  They are thin
                    # (width 1.0) so they would fall into "internal" without this
                    # check.  Pattern is constant across all Sivagangai fixtures.
                    for s, e in segs:
                        chain.append(Segment(start=s, end=e, width=width))
                elif stroke_color == "blue":
                    # Blue strokes: originally thought to be chain lines but are
                    # actually chain-arrow markers in some older district PDFs.
                    # Keep routing here for backwards compatibility.
                    for s, e in segs:
                        chain.append(Segment(start=s, end=e, width=width))
                elif width >= BOUNDARY_MIN_WIDTH:
                    # Thick black path: boundary or separation
                    total = sum(math.hypot(e[0] - s[0], e[1] - s[1]) for s, e in segs)
                    if total >= threshold_pt:
                        # Long path -> main boundary edge
                        for s, e in segs:
                            boundary.append(Segment(start=s, end=e, width=width))
                    else:
                        # Short thick path: connector or separation tick
                        both_on_bnd = all(
                            (round(s[0]), round(s[1])) in bnd_pts
                            and (round(e[0]), round(e[1])) in bnd_pts
                            for s, e in segs
                        )
                        if both_on_bnd:
                            for s, e in segs:
                                boundary.append(Segment(start=s, end=e, width=width))
                        else:
                            for s, e in segs:
                                separation.append(Segment(start=s, end=e, width=width))
                else:
                    # Thin black path: subdivision or boundary duplicate.
                    # Only filter if the segment's endpoint PAIR exactly matches a
                    # thick boundary segment — connecting boundary corners across the
                    # plot interior is a valid subdivision line, not a duplicate.
                    is_dup = bool(segs) and all(
                        frozenset(((round(s[0]), round(s[1])), (round(e[0]), round(e[1]))))
                        in bnd_pairs
                        for s, e in segs
                    )
                    if not is_dup:
                        for s, e in segs:
                            internal.append(Segment(start=s, end=e, width=width))

            if has_fill:
                fill_color = _color_name(drawing.get("fill"))
                if fill_color == "blue":
                    blue_markers.append(_to_marker(drawing))
                elif fill_color == "red":
                    stones.append(_to_marker(drawing))

        # --- Reroute separation ticks misclassified as boundary ----------------
        # The FMB draws a short stub at some boundary corners, extending OUTWARD
        # toward the adjacent survey (the "separation" mark).  These stubs are
        # encoded inside the same thick-black polylines as the real boundary, so
        # the long-path test files them as boundary; polygonize then drops them
        # from the ring (they enclose no face), and they vanish entirely.
        #
        # Detect them by graph degree over the boundary endpoints: a separation
        # tick has one DANGLING endpoint (degree 1) whose other end sits on a
        # boundary vertex the ring passes through (degree >= 3 = two ring edges
        # plus the stub).  An honestly-open boundary's loose ends instead branch
        # off a degree-2 vertex, so they are NOT rerouted.
        _bnd_deg: dict[tuple[int, int], int] = {}
        for _s in boundary:
            _bnd_deg[(round(_s.start[0]), round(_s.start[1]))] = (
                _bnd_deg.get((round(_s.start[0]), round(_s.start[1])), 0) + 1
            )
            _bnd_deg[(round(_s.end[0]), round(_s.end[1]))] = (
                _bnd_deg.get((round(_s.end[0]), round(_s.end[1])), 0) + 1
            )
        _kept_boundary: list[Segment] = []
        for _s in boundary:
            _a = _bnd_deg[(round(_s.start[0]), round(_s.start[1]))]
            _b = _bnd_deg[(round(_s.end[0]), round(_s.end[1]))]
            if (_a == 1 and _b >= 3) or (_b == 1 and _a >= 3):
                separation.append(_s)
            else:
                _kept_boundary.append(_s)
        boundary = _kept_boundary

        # --- Glyph extraction + classification (vector-first architecture) -----
        # Cluster the colored filled paths into per-word groups and classify by
        # color + area.  This runs in the same open-document context as the line
        # extraction, so no second fitz.open() is needed.  The resulting groups
        # carry exact PDF positions and a semantic kind — OCR only reads values.
        from .pdf_glyphs import extract_glyph_groups  # noqa: PLC0415

        glyph_groups = extract_glyph_groups(page)
        _classify_glyph_groups(glyph_groups, pw, ph)

    result = PageVectors(
        boundary=boundary,
        internal=internal,
        separation=separation,
        chain=chain,
        blue_markers=blue_markers,
        stones=stones,
        dashed_ref=dashed_ref,
        glyph_groups=glyph_groups,
        page_width=pw,
        page_height=ph,
    )
    c = result.counts()
    by_kind: dict[str, int] = {}
    for g in glyph_groups:
        by_kind[g.kind] = by_kind.get(g.kind, 0) + 1
    _log.info(
        "vectors [%s p%d]: page=%.0fx%.0f bnd=%d int=%d sep=%d chain=%d "
        "stones=%d blue=%d dashed_ref=%d glyphs=%d %s",
        path.name, page_number,
        pw, ph,
        c["boundary"], c["internal"], c["separation"],
        c["chain"], c["stones"], c["blue_markers"], c["dashed_ref"],
        len(glyph_groups),
        " ".join(f"{k}={v}" for k, v in sorted(by_kind.items())),
    )
    return result


def _classify_glyph_groups(groups: list, pw: float, ph: float) -> None:
    """Classify glyph groups in-place by color and bbox area.

    Blue groups (dimension numbers and parcel/survey labels):
      * The single largest blue cluster by bbox area is the survey number.
      * Blue clusters with area > 3× the median are parcel labels (2A, 3B, …).
      * Remaining blue clusters are dimension values (distances, chain lengths).

    Red groups: all ``"red_label"`` — OCR downstream tells stone letters (A–H)
    from neighbor survey numbers (96, 99, 101) by value + position.

    Black groups: ``"neighbor"`` when the cluster centre is within _NEIGHBOR_EDGE_PT
    of any page edge; otherwise ``"black_label"``.
    """
    _NEIGHBOR_EDGE_PT = 80.0

    # Blue groups: the survey-number glyph (e.g. "100") is drawn at ~20× the
    # size of any dimension label.  Use an absolute threshold of 5000 pt² so
    # that rotated dimension labels (whose AXIS-ALIGNED bbox expands at steep
    # angles) are never mis-promoted.  Everything else is "dimension" —
    # parcel labels (2A, 3B) and pure integers are filtered downstream by
    # _NON_MEASUREMENT_RE and _SUB_PLOT_RE in anchor.py.
    _SURVEY_NUM_AREA_PT2 = 5000.0

    for g in groups:
        if g.color == "blue":
            if g.bbox.get_area() >= _SURVEY_NUM_AREA_PT2:
                g.kind = "survey_number"
            else:
                g.kind = "dimension"

    for g in groups:
        if g.color == "red":
            g.kind = "red_label"
        elif g.color == "black":
            cx, cy = g.center
            if (cx < _NEIGHBOR_EDGE_PT or cx > pw - _NEIGHBOR_EDGE_PT
                    or cy > ph - _NEIGHBOR_EDGE_PT):
                g.kind = "neighbor"
            else:
                g.kind = "black_label"
