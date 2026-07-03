"""Assemble classified vectors + anchored measurements + header into a Plot.

Single responsibility: the M1 assembly point. Take everything the earlier steps
produced and turn it into one :class:`~landintel.core.models.Plot` in
**real-world metres**. Pure assembly -- it computes, it does not judge.

Two things this module is careful about:

* **Scale is applied once, uniformly.** Every pixel coordinate -- boundary ring
  and corner points alike -- goes through the same transform built from the
  per-PDF scale, so nothing downstream ever sees mixed units. Without a scale
  there is no way to produce real-world geometry, so a missing scale is a hard
  :class:`GeometryError` rather than a silent fallback to pixel units.

* **The area cross-check is set up, not decided.** The Plot carries both the
  geometry-derived area (via ``Boundary.computed_area``, in m^2 because the
  points are in metres) and the header's stated area (``stated_area``, in
  hectares). This module makes both available; the anomaly layer (step 10) reads
  them and decides flag-vs-pass. Compute and judge stay separate -- the same way
  normalization stays out of ``ocr.py``.

Boundary construction: the thick-black segments form a planar subdivision (an
outer perimeter plus internal shared edges), not a tidy cycle, so the perimeter
is recovered with Shapely ``polygonize`` -- which nodes the network and finds the
faces -- and the union's exterior is the ring. If the segments do not enclose any
face (a real, missing-edge plot), the boundary is left **open**: a best-effort
endpoint chain whose ``Boundary.is_closed`` is honestly ``False`` for the anomaly
layer to flag. No forced closure that would fake a valid area.
"""

from __future__ import annotations

import logging
import math
import re

from shapely.geometry import LineString, Polygon
from shapely.geometry import Point as ShapelyPoint

_log = logging.getLogger(__name__)
from shapely.ops import polygonize, unary_union

from ...core.enums import MeasurementSource, PlotStatus
from ...core.exceptions import GeometryError
from ...core.models import Boundary, CornerPoint, Measurement, NeighborLabel, Plot, Point, SubPlotLabel
from .anchor import AnchorResult
from .ocr import FmbHeader, OCRDetection
from .pdf_vectors import BOUNDARY_MIN_WIDTH, Marker, PageVectors, Segment

__all__ = ["build_plot", "points_to_metres"]

# Client standard: separation "whisker" lines (corner -> neighbour) are drawn at a
# fixed 21 m length on every FMB, regardless of the arbitrary length in the source.
SEPARATION_LEN_M = 21.0

# A subdivision-line endpoint within this distance (m) of the boundary is treated as a
# gap to close and snapped onto the boundary ring (client: "subdivision lines not
# attached to boundary"). Small enough not to disturb genuine interior T-junctions.
SUBDIV_SNAP_TOL_M = 3.0
# A DANGLING subdivision end (an open end shared with no other subdivision segment) that
# should reach the boundary but stops short is snapped within this larger tolerance --
# only open ends move, so interior T-junctions (shared, degree >= 2) are never distorted.
SUBDIV_DANGLE_TOL_M = 8.0

# Endpoint coincidence tolerance (PDF points) when chaining an open boundary.
_CHAIN_TOLERANCE = 0.5

# Max distance (PDF points) to borrow a stone's label from a nearby OCR token.
_LABEL_RADIUS = 16.0


def points_to_metres(scale_denominator: int) -> float:
    """Metres of real-world length per PDF point of drawing, for this scale.

    point -> inch (/72) -> cm (x2.54) -> apply scale -> metres (/100).
    """
    return (2.54 / 72.0) * scale_denominator / 100.0


def _transform(point: Point, ppm: float, page_height: float) -> Point:
    """Pixel point -> real-world metres, flipping y to a conventional y-up frame."""
    x, y = point
    return (x * ppm, (page_height - y) * ppm)


_MEAS_VALUE_RE = re.compile(r"\d{1,4}[.,]\d{1,2}")


def _parse_meas_value(text: str) -> float | None:
    """Parse a measurement's numeric value (e.g. '44,2' / '(160.2)' -> 44.2 / 160.2)."""
    m = _MEAS_VALUE_RE.search(text.strip().strip("()"))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None


def _drop_boundary_totals(
    anchored: list,
    ppm: float,
    *,
    boundary_segments: list | None = None,
    ratio_thresh: float = 1.4,
    angle_tol_deg: float = 12.0,
    perp_tol_m: float = 6.0,
) -> list:
    """Remove boundary EDGE TOTALS (the span value = sum of its segments).

    An FMB prints both the per-segment boundary values (45, 55, 60 …) and the
    sum of them across the whole corner-to-corner edge (160).  We drop only that
    sum, and only on boundary lines.  A candidate is any boundary value that is
    ``> ratio_thresh x its own anchored line length`` (condition 1 — a real
    segment value ~equals its own line, so a span total stands out).  It is then
    confirmed a total if EITHER:

      2a. VECTOR span (recall-independent, primary): the value ≈ the metric length
          of the collinear boundary RUN it sits on, summed from the ``boundary_segments``
          VECTOR lengths — no dependence on the neighbour values being OCR'd.
      2b. OCR sum (the original signal): value ≈ sum of >=2 OTHER collinear boundary
          VALUES each smaller than it.

    0-FP: a real segment value equals its OWN short line, so it fails condition 1 and is never a
    candidate; only a genuine span total reaches 2a/2b and matches the full-run length.  Before,
    at ~24% OCR recall the segments were rarely all read so 2b missed most totals; 2a fixes that
    from the vector geometry.
    """
    items = [(am, _parse_meas_value(am.text)) for am in anchored]

    def seg_angle(line) -> float:
        return math.degrees(
            math.atan2(line.end[1] - line.start[1], line.end[0] - line.start[0])
        ) % 180.0

    def perp_mid_dist(line2, line) -> float:
        mx = (line2.start[0] + line2.end[0]) / 2.0
        my = (line2.start[1] + line2.end[1]) / 2.0
        dx, dy = line.end[0] - line.start[0], line.end[1] - line.start[1]
        seg = math.hypot(dx, dy)
        if seg < 1e-9:
            return math.hypot(mx - line.start[0], my - line.start[1])
        return abs((mx - line.start[0]) * dy - (my - line.start[1]) * dx) / seg

    def _vector_run_len_m(anchor_line) -> float:
        """Metric length of the collinear boundary RUN the anchor sits on, from VECTOR
        segments (all boundary strokes, not just the OCR'd ones)."""
        if not boundary_segments:
            return 0.0
        a0 = seg_angle(anchor_line)
        total_px = 0.0
        for s in boundary_segments:
            ad = abs(((seg_angle(s) - a0 + 90.0) % 180.0) - 90.0)
            if ad > angle_tol_deg:
                continue
            if perp_mid_dist(s, anchor_line) * ppm > perp_tol_m:
                continue
            total_px += math.hypot(s.end[0] - s.start[0], s.end[1] - s.start[1])
        return total_px * ppm

    drop: set[int] = set()
    for i, (am, val) in enumerate(items):
        if am.line_class != "boundary" or val is None or val <= 0:
            continue
        length_m = math.hypot(
            am.line.end[0] - am.line.start[0], am.line.end[1] - am.line.start[1]
        ) * ppm
        if length_m < 1e-6 or val <= ratio_thresh * length_m:
            continue  # condition 1: must be much longer than its anchored line
        # 2a. VECTOR span (recall-independent): value ~= the full collinear boundary run length.
        run_m = _vector_run_len_m(am.line)
        if run_m > ratio_thresh * length_m and abs(run_m - val) <= max(2.0, 0.08 * val):
            drop.add(i)
            _log.info("drop boundary total (vector run): %s ~= %.1f m span", am.text, run_m)
            continue
        # 2b. OCR sum (original): value ~= sum of >=2 collinear neighbour VALUES.
        a0 = seg_angle(am.line)
        neighbours: list[float] = []
        for j, (am2, v2) in enumerate(items):
            if j == i or am2.line_class != "boundary" or v2 is None or v2 <= 0 or v2 >= val:
                continue
            ad = abs(((seg_angle(am2.line) - a0 + 90.0) % 180.0) - 90.0)
            if ad > angle_tol_deg:
                continue
            if perp_mid_dist(am2.line, am.line) * ppm > perp_tol_m:
                continue
            neighbours.append(v2)
        if len(neighbours) >= 2 and abs(sum(neighbours) - val) <= max(2.0, 0.05 * val):
            drop.add(i)
            _log.info(
                "drop boundary total (OCR sum): %s = Σ%s", am.text,
                "+".join(f"{n:g}" for n in sorted(neighbours)),
            )
    return [am for k, (am, _) in enumerate(items) if k not in drop]


def _simplify_collinear(ring: list[Point], tol_deg: float = 2.0) -> list[Point]:
    """Drop near-collinear (straight-through) vertices from a closed ring.

    A straight boundary line split across several collinear vertices renders as
    several polylines; removing the in-line vertices (bend < ``tol_deg``) merges
    each straight run into ONE edge → one polyline per boundary line.

    ``tol_deg`` is deliberately small (2°): only true polygonize artifacts (two
    exactly-collinear source segments sharing a vertex, bend ~0-1°) are merged.
    Genuine slight corners — e.g. survey 31's real 5.4° corner, survey 100's
    6-7° bends — are KEPT, so the area cross-check is unaffected.  Returns the
    ring unchanged if it would collapse below a triangle.
    """
    if len(ring) < 4:
        return ring
    pts = ring[:-1] if ring[0] == ring[-1] else list(ring)
    n = len(pts)
    kept: list[Point] = []
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2)
        if math.degrees(math.asin(min(1.0, cross))) >= tol_deg:
            kept.append(b)  # a real corner
    if len(kept) < 3:
        return ring
    return kept + [kept[0]]


def _greedy_chain(segments: list[Segment]) -> list[Point]:
    """Order disconnected segments into a single open polyline, best effort.

    Used only when the segments enclose no face. Starts at a loose end (degree-1
    endpoint) when there is one, then walks unused segments by endpoint
    coincidence. The result is intentionally allowed to stay open.
    """
    if not segments:
        return []

    def key(p: Point) -> tuple[int, int]:
        return (round(p[0] / _CHAIN_TOLERANCE), round(p[1] / _CHAIN_TOLERANCE))

    incident: dict[tuple[int, int], list[tuple[int, Point, Point]]] = {}
    for index, seg in enumerate(segments):
        incident.setdefault(key(seg.start), []).append((index, seg.start, seg.end))
        incident.setdefault(key(seg.end), []).append((index, seg.end, seg.start))

    # Prefer to start at a loose end so the whole chain is traversed in order.
    start_pt = segments[0].start
    for entries in incident.values():
        if len(entries) == 1:
            start_pt = entries[0][1]
            break

    used: set[int] = set()
    points: list[Point] = [start_pt]
    current = start_pt
    while True:
        nxt: tuple[int, Point] | None = None
        for index, here, other in incident.get(key(current), []):
            if index not in used:
                nxt = (index, other)
                break
        if nxt is None:
            break
        index, other = nxt
        used.add(index)
        points.append(other)
        current = other
    return points


def _boundary_ring(
    segments: list[Segment],
    bridge_segments: list[Segment] | None = None,
    internal_segments: list[Segment] | None = None,
) -> tuple[list[Point], bool]:
    """Recover the boundary as ordered pixel points; report whether it closed.

    Returns ``(points, closed)``. ``closed`` is ``True`` when the segments
    enclosed a face (perimeter via ``polygonize``); ``False`` when they did not
    (open best-effort chain).

    ``bridge_segments`` (the separation + dashed-reference strokes) are a
    fallback used ONLY when the solid boundary fails to enclose a face. Some
    districts (e.g. INGUR/Erode) draw a parcel's natural or shared boundary edge
    DASHDOT rather than solid; pdf_vectors files that thick-but-dashed edge under
    SEPARATION, so it is absent from ``segments`` and the perimeter cannot close.
    When that happens we retry polygonize with the thick (>= BOUNDARY_MIN_WIDTH)
    bridge strokes added — just enough to complete the loop. The union-of-faces
    step then dissolves any internal separators back into the single outer
    perimeter. Plots whose solid boundary already closes never reach this branch,
    so the Sivagangai fixtures (all of which close) are unaffected.

    ``internal_segments`` is a LAST-RESORT fallback for subdivision-dense sheets
    (e.g. Manur survey 112) that draw the OUTER parcel edge at the same stroke
    weight as the internal subdivision lines, so pdf_vectors classifies the whole
    outline as internal and ``segments`` is empty/non-enclosing. polygonizing the
    boundary+internal network and dissolving the sub-faces (``unary_union``) yields
    the outer parcel exterior — the internal subdivisions dissolve to interior
    edges. Only reached when thick boundary AND the dashed bridge both fail to
    enclose a face, so plots with a genuine thick boundary are unaffected.
    """
    base_lines = [LineString([s.start, s.end]) for s in segments]
    polygons = list(polygonize(unary_union(base_lines))) if base_lines else []

    if not polygons and bridge_segments:
        thick_bridge = [
            LineString([s.start, s.end])
            for s in bridge_segments
            if getattr(s, "width", 0.0) >= BOUNDARY_MIN_WIDTH
        ]
        if thick_bridge:
            polygons = list(polygonize(unary_union(base_lines + thick_bridge)))
            if polygons:
                _log.info(
                    "Boundary closed by borrowing %d thick separation/dashed edge(s) "
                    "(solid boundary alone did not enclose a face)",
                    len(thick_bridge),
                )

    if not polygons and internal_segments:
        all_lines = base_lines + [
            LineString([s.start, s.end]) for s in internal_segments
        ]
        polygons = list(polygonize(unary_union(all_lines)))
        if polygons:
            _log.info(
                "Boundary recovered from the boundary+internal network (%d internal "
                "edges); no thick boundary stroke enclosed a face (subdivision-dense "
                "sheet drawn at uniform stroke weight)",
                len(internal_segments),
            )

    if polygons:
        merged = unary_union(polygons)
        if merged.geom_type == "MultiPolygon":
            # Disconnected pieces: the largest face is the plot's perimeter.
            merged = max(merged.geoms, key=lambda g: g.area)
        ring = [(float(x), float(y)) for x, y in merged.exterior.coords]
        return ring, True

    return _greedy_chain(segments), False


_CORNER_LETTER_RE = re.compile(r"^[A-H]$")
_STONE_ID_RE      = re.compile(r"^\d{1,2}$")


def _is_valid_stone_label(text: str) -> bool:
    """Valid stone labels: corner letters A-H, or integers 1-31."""
    t = text.strip()
    if _CORNER_LETTER_RE.match(t):
        return True
    if _STONE_ID_RE.match(t):
        return 1 <= int(t) <= 31
    return False


def _nearest_label(
    marker: Marker,
    detections: list[OCRDetection],
    used: set[str],
) -> str:
    """Borrow a stone's label from the closest valid, unused OCR token."""
    best_label = ""
    best_dist = _LABEL_RADIUS
    for det in detections:
        token = det.text.strip()
        if not _is_valid_stone_label(token):
            continue
        if token in used:
            continue
        cx, cy = det.center
        dist = math.hypot(cx - marker.x, cy - marker.y)
        if dist < best_dist:
            best_dist = dist
            best_label = token
    return best_label


# Max distance (PDF points) a stone glyph may be from a line junction to snap to
# it.  Stones measured at 0.7-9.5 m off their junction, so ~14 m (≈ the largest
# observed gap with margin) snaps every real stone while leaving a truly stray
# red fill where it is.
_STONE_SNAP_MAX_PT = 22.0


def _connecting_points(vectors: PageVectors, ring_px: list[Point]) -> list[Point]:
    """Real survey corners a stone can sit on: boundary ring vertices + the
    endpoints of boundary and subdivision (internal) lines.

    CHAIN/traverse endpoints are deliberately EXCLUDED: the traverse is a dashed
    overlay whose many little dash endpoints are scattered along its length, so
    including them made interior stones snap to a nearby dash point instead of
    their actual line junction.  Every stone has a boundary/subdivision corner
    within snapping range, so excluding chain loses nothing and fixes the scatter.
    """
    pts: list[Point] = list(ring_px)
    for seg in list(vectors.internal) + list(vectors.boundary):
        pts.append(seg.start)
        pts.append(seg.end)
    return pts


def _snap_stone(marker: Marker, connecting: list[Point]) -> Point:
    """Snap a stone to its nearest connecting point (the line junction it marks).

    Returns the junction in PDF points — the stone's true coordinate.  If no
    junction is within ``_STONE_SNAP_MAX_PT`` the stone is left where it is (a
    stray red fill, not a real corner).
    """
    gx, gy = marker.x, marker.y
    best: Point | None = None
    best_d = _STONE_SNAP_MAX_PT
    for cx, cy in connecting:
        d = math.hypot(cx - gx, cy - gy)
        if d < best_d:
            best_d = d
            best = (cx, cy)
    return best if best is not None else (gx, gy)


def _line_angle_dxf(segment: Segment, page_height: float) -> float:
    """Angle of segment in DXF y-up space, normalised to (-90, 90] for upright text.

    PDF y increases downward; DXF y increases upward after the y-flip in
    ``_transform``.  Negating dy converts between the two frames before
    calling atan2.  A DXF *text rotation* in [90, 180) renders the label
    nearly upside-down / mirror-flipped (e.g. a top edge at 176° read as
    "9'St7L" instead of "145,6").  Normalising to (-90, 90] — a 180° flip is
    the same line direction, just readable — keeps every label upright, exactly
    as the FMB sheet and the client's manual DXF draw them.
    """
    dx = segment.end[0] - segment.start[0]
    dy_pdf = segment.end[1] - segment.start[1]
    dy_dxf = -dy_pdf  # flip y axis to match DXF convention
    a = math.degrees(math.atan2(dy_dxf, dx)) % 180.0
    if a > 90.0:
        a -= 180.0
    return a


def _glyph_angle_to_dxf(angle_deg: float | None) -> float | None:
    """Convert a glyph baseline angle (PDF space, [0,180)) to a DXF text rotation.

    The build flips y (PDF y-down → DXF y-up), so a PDF baseline at ``a``
    becomes ``180 - a`` in DXF.  Normalise to (-90, 90] so the label is never
    rendered upside-down / mirror-flipped (a 180° text flip is the same line
    direction, just unreadable).
    """
    if angle_deg is None:
        return None
    r = (180.0 - angle_deg) % 180.0
    if r > 90.0:
        r -= 180.0
    return r


def _line_ref(segment: Segment) -> str:
    """A stable identifier for the line a measurement labels (its midpoint)."""
    mx = (segment.start[0] + segment.end[0]) / 2.0
    my = (segment.start[1] + segment.end[1]) / 2.0
    return f"{mx:.0f},{my:.0f}"


# Uniform label-to-line gap as a fraction of plot extent (min 2 m floor).
# Every dimension label is held at exactly this perpendicular distance from its
# line so spacing is identical across the whole map and no label sits on a line.
# Extent-based (~3.9 m on survey 100) is the client-preferred spacing — it gives
# the label clear breathing room off its line rather than hugging it tight.
_LABEL_OFFSET_EXTENT_FRACTION: float = 0.018
_LABEL_OFFSET_MIN_M: float = 2.0


def _point_segment_dist(p: Point, a: Point, b: Point) -> float:
    """Shortest distance from point ``p`` to segment ``a-b`` (metre space)."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _snap_label_off_line(
    label_pos: Point,
    line_start: Point,
    line_end: Point,
    offset_m: float,
    outward_from: Point | None = None,
) -> Point:
    """Project a label onto its line, then offset perpendicular by ``offset_m``.

    The along-line position is the nearest point on the segment to the OCR label
    (which section of the line the number annotates).  The perpendicular offset
    is a fixed ``offset_m`` so every label sits the SAME distance from its line,
    never on it.

    Side selection:
    * ``outward_from`` given (the plot centroid) -> the label is placed on the
      side of the line AWAY from that point.  Used for boundary dimensions so
      they always sit OUTSIDE the plot, never inside it.
    * otherwise -> the side OCR found the glyph on (preserves layout for
      internal / chain labels, which belong inside the plot).
    """
    lx, ly = label_pos
    ax, ay = line_start
    bx, by = line_end
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return (lx, ly + offset_m)

    t = max(0.0, min(1.0, ((lx - ax) * dx + (ly - ay) * dy) / seg_len_sq))
    px, py = ax + t * dx, ay + t * dy

    seg_len = math.sqrt(seg_len_sq)
    perp_x = -dy / seg_len
    perp_y = dx / seg_len

    if outward_from is not None:
        # Point the offset AWAY from the plot centroid (outside the boundary).
        if (px - outward_from[0]) * perp_x + (py - outward_from[1]) * perp_y < 0.0:
            perp_x, perp_y = -perp_x, -perp_y
    else:
        # Keep the label on the side OCR found it (cross/dot sign).
        if (lx - px) * perp_x + (ly - py) * perp_y < 0.0:
            perp_x, perp_y = -perp_x, -perp_y

    return (px + perp_x * offset_m, py + perp_y * offset_m)


def _snap_to_nearest_chain(
    label_pos: Point,
    chain_segs: list[tuple[Point, Point]],
    offset_m: float,
) -> Point:
    """Snap a chain-dim label to its nearest chain line, offset perpendicular."""
    best_dist = math.inf
    best_seg: tuple[Point, Point] | None = None
    for start, end in chain_segs:
        d = _point_segment_dist(label_pos, start, end)
        if d < best_dist:
            best_dist = d
            best_seg = (start, end)
    if best_seg is None:
        return label_pos
    return _snap_label_off_line(label_pos, best_seg[0], best_seg[1], offset_m)


def _label_aabb(
    pos: Point, angle_deg: float, text: str, height: float, padding: float = 0.5,
) -> tuple[float, float, float, float]:
    """Axis-aligned bbox (min_x, min_y, max_x, max_y) of a rotated text label."""
    w = len(text) * height * 0.6 + padding * 2
    h = height + padding * 2
    hw, hh = w / 2, h / 2
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    cx, cy = pos
    xs, ys = [], []
    for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        xs.append(cx + x * ca - y * sa)
        ys.append(cy + x * sa + y * ca)
    return min(xs), min(ys), max(xs), max(ys)


def _aabb_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float],
) -> bool:
    """True if two axis-aligned bounding boxes overlap."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _resolve_label_overlaps(
    measurements: list[Measurement],
    height: float,
    min_gap_m: float = 1.0,
    max_iters: int = 12,
) -> None:
    """Separate genuinely-overlapping labels in-place, conservatively.

    Each overlapping label is slid ALONG its own line direction (``line_angle``)
    away from its neighbour, so it stays at the same perpendicular offset from
    its line — it just shifts to a less-crowded spot along that line.  This keeps
    every label on its line (unlike a center-to-center push, which would drift
    labels off) and only acts on true overlaps, so the result stays at the
    manual's label density rather than spreading wider.
    """
    n = len(measurements)
    if n <= 1:
        return
    pos = [list(m.position) for m in measurements if m.position]
    if len(pos) != n:
        return  # some label has no position; skip rather than misalign indices
    ang = [m.line_angle if m.line_angle is not None else 0.0 for m in measurements]
    txt = [m.raw for m in measurements]

    for _ in range(max_iters):
        moved = False
        for i in range(n):
            bi = _label_aabb((pos[i][0], pos[i][1]), ang[i], txt[i], height)
            for j in range(i + 1, n):
                bj = _label_aabb((pos[j][0], pos[j][1]), ang[j], txt[j], height)
                if not _aabb_overlap(bi, bj):
                    continue
                moved = True
                ox = min(bi[2], bj[2]) - max(bi[0], bj[0])
                oy = min(bi[3], bj[3]) - max(bi[1], bj[1])
                push = (max(min(ox, oy), 0.1) + min_gap_m) / 2.0
                # Slide each label along its own line, away from the other.
                for k, other in ((i, j), (j, i)):
                    rad = math.radians(ang[k])
                    dx, dy = math.cos(rad), math.sin(rad)
                    sep = (pos[k][0] - pos[other][0]) * dx + (pos[k][1] - pos[other][1]) * dy
                    s = 1.0 if sep >= 0 else -1.0
                    pos[k][0] += dx * push * s
                    pos[k][1] += dy * push * s
        if not moved:
            break

    for i, m in enumerate(measurements):
        m.position = (pos[i][0], pos[i][1])



def build_plot(
    *,
    client_id: str,
    vectors: PageVectors,
    detections: list[OCRDetection],
    anchor_result: AnchorResult,
    header: FmbHeader,
) -> Plot:
    """Build one :class:`Plot` in real-world metres from M1's outputs.

    Args:
        client_id: Owning tenant.
        vectors: Classified segments and markers from ``pdf_vectors``.
        detections: All OCR detections (used to label corner stones).
        anchor_result: Measurement-to-line pairings from ``anchor``.
        header: Parsed header metadata, including the scale and stated area.

    Returns:
        A ``Plot`` with scaled boundary, corner points and measurements, plus the
        stated area, ready for the agent layer to validate.

    Raises:
        GeometryError: If the header carried no scale -- real-world geometry
            cannot be produced and emitting pixel units mislabelled as metres
            would silently corrupt M2/M3.
    """
    if header.scale_denominator is None:
        raise GeometryError(
            "cannot build plot without a scale", survey_no=header.survey_no
        )

    # Header scale is the honest best estimate: measurements imply a ppm within
    # ~0.5% of it, so the few-metre value-vs-length gap is drawing/OCR precision,
    # not a scale error — empirical calibration only adds noise (and worsened the
    # area match on survey 100), so we keep the exact header scale.
    ppm = points_to_metres(header.scale_denominator)
    page_height = vectors.page_height

    ring_px, _closed = _boundary_ring(
        vectors.boundary,
        bridge_segments=list(vectors.separation) + list(vectors.dashed_ref),
        internal_segments=list(vectors.internal),
    )
    # Merge collinear runs so each straight boundary line is ONE polyline (not
    # several collinear ones).  The FULL ring_px is kept below for stone
    # connecting points; only the drawn/area boundary uses the simplified ring.
    boundary = Boundary(
        points=[_transform(p, ppm, page_height) for p in _simplify_collinear(ring_px)]
    )
    # Plot centroid (metres) — boundary dimension labels are pushed to the side
    # AWAY from this so they always sit outside the plot.
    if boundary.points:
        plot_centroid: Point | None = (
            sum(p[0] for p in boundary.points) / len(boundary.points),
            sum(p[1] for p in boundary.points) / len(boundary.points),
        )
    else:
        plot_centroid = None

    # Dimension text height (matches to_dxf.text_height_for) for collision bbox
    # sizing, plus the uniform label-to-line offset (extent-based) that every
    # measurement label sits at off its line.
    if ring_px:
        xs_px = [p[0] for p in ring_px]
        ys_px = [p[1] for p in ring_px]
        extent_m = max(max(xs_px) - min(xs_px), max(ys_px) - min(ys_px)) * ppm
        label_height = max(1.5, min(6.0, extent_m * 0.014))
        label_offset = max(_LABEL_OFFSET_MIN_M, extent_m * _LABEL_OFFSET_EXTENT_FRACTION)
    else:
        label_height = 3.0
        label_offset = _LABEL_OFFSET_MIN_M
    _log.info(
        "build_plot [survey=%s]: scale=1:%d ppm=%.6f bnd_pts=%d closed=%s "
        "stones=%d measurements=%d stated_area=%.4fha",
        header.survey_no, header.scale_denominator, ppm,
        len(boundary.points), _closed,
        len(vectors.stones), len(anchor_result.anchored),
        header.stated_area_ha or 0.0,
    )

    # Stones snap to the connecting points (line junctions) they mark, so the
    # coordinate IS the corner M2 needs.  to_dxf LEFT-justifies the number at this
    # point, putting the text grip on the line-endpoint node.
    connecting = _connecting_points(vectors, ring_px)

    used_stone_labels: set[str] = set()
    corner_points = []
    for stone in vectors.stones:
        label = _nearest_label(stone, detections, used_stone_labels)
        if label:
            used_stone_labels.add(label)
        cx, cy = _transform(_snap_stone(stone, connecting), ppm, page_height)
        corner_points.append(CornerPoint(label=label, x=cx, y=cy))

    # Drop boundary edge totals (e.g. 160 = 45+55+60) — boundary lines only. The VECTOR-run
    # check (boundary_segments) catches totals even when the segment values were not all OCR'd.
    anchored = _drop_boundary_totals(anchor_result.anchored, ppm,
                                     boundary_segments=list(vectors.boundary))

    measurements = [
        Measurement(
            raw=m.text,
            value=None,  # normalization is the validator's job, not assembly's
            source=MeasurementSource.OCR,
            confidence=m.confidence,
            line_ref=_line_ref(m.line),
            line_class=m.line_class,
            # Snap-to-line placement: project onto the anchor line at the OCR
            # along-line position, then offset by a uniform gap so every label
            # sits the SAME distance off its line, never on it.  Boundary dims
            # are forced OUTSIDE the plot (away from the centroid); internal /
            # chain dims keep the OCR side (they belong inside).
            position=_snap_label_off_line(
                _transform(m.center, ppm, page_height),
                _transform(m.line.start, ppm, page_height),
                _transform(m.line.end, ppm, page_height),
                label_offset,
                outward_from=plot_centroid if m.line_class == "boundary" else None,
            ),
            line_length_m=math.hypot(
                m.line.end[0] - m.line.start[0], m.line.end[1] - m.line.start[1]
            )
            * ppm,
            line_angle=_line_angle_dxf(m.line, page_height),
        )
        for m in anchored
    ]

    # Boundary ring in real metres -- shared by the subdivision-attach snap AND the
    # separation-whisker normalisation below.
    _ring_m = [_transform(p, ppm, page_height) for p in ring_px] if len(ring_px) >= 2 else []
    _bnd_ring_m = Polygon(_ring_m).exterior if len(_ring_m) >= 4 else None

    # Transform subdivision segments to metres first, then compute endpoint DEGREE (how
    # many segment-ends coincide there) so a DANGLING open end (degree 1) can be attached
    # with a wider reach than a shared interior node.
    _sub_raw = [(_transform(s.start, ppm, page_height), _transform(s.end, ppm, page_height))
                for s in vectors.internal]
    _sub_eps = [ep for seg in _sub_raw for ep in seg]

    def _sub_degree(pt: Point) -> int:
        return sum(1 for q in _sub_eps if math.hypot(pt[0] - q[0], pt[1] - q[1]) < 0.3)

    def _snap_to_boundary(pt: Point) -> Point:
        """Attach a subdivision endpoint to the boundary when it stops short of it.
        A subdivision line divides the parcel, so its ends lie ON the boundary; vector
        extraction sometimes leaves a gap. An endpoint within SUBDIV_SNAP_TOL_M snaps; a
        DANGLING open end (shared with no other subdivision segment) snaps within the wider
        SUBDIV_DANGLE_TOL_M -- so an open end reaches the boundary while interior T-junctions
        (shared, degree >= 2) stay put and the internal network is never distorted."""
        if _bnd_ring_m is None:
            return pt
        p = ShapelyPoint(pt[0], pt[1])
        tol = SUBDIV_DANGLE_TOL_M if _sub_degree(pt) <= 1 else SUBDIV_SNAP_TOL_M
        if _bnd_ring_m.distance(p) <= tol:
            proj = _bnd_ring_m.interpolate(_bnd_ring_m.project(p))
            return (proj.x, proj.y)
        return pt

    subdivision_segments = [(_snap_to_boundary(a), _snap_to_boundary(b)) for a, b in _sub_raw]
    # Chain/traverse: keep only segments INSIDE the plot boundary.  The traverse
    # direction arrows (the ">>>" chevrons) and neighbour-connection legs sit
    # OUTSIDE the boundary — they are noise for a single-plot extraction.  An
    # internal traverse connecting this plot's own stones stays inside.
    _bnd_poly = Polygon(ring_px) if len(ring_px) >= 4 else None
    _bnd_poly_buf = _bnd_poly.buffer(3.0) if _bnd_poly is not None else None

    def _chain_inside(seg: Segment) -> bool:
        if _bnd_poly_buf is None:
            return True
        mx = (seg.start[0] + seg.end[0]) / 2.0
        my = (seg.start[1] + seg.end[1]) / 2.0
        return _bnd_poly_buf.contains(ShapelyPoint(mx, my))

    chain_segments = [
        (_transform(s.start, ppm, page_height), _transform(s.end, ppm, page_height))
        for s in vectors.chain
        if _chain_inside(s)
    ]
    # Separation "whisker" lines stick OUT from the plot's corner stones toward the
    # neighbouring parcels; the length actually drawn is arbitrary (it runs to the
    # neighbour or the sheet edge). Client standard: a FIXED 21 m whisker on every
    # FMB. Keep each whisker's boundary-anchored end + its direction, and normalise
    # the length to exactly SEPARATION_LEN_M.
    def _norm_separation(a: Point, b: Point) -> tuple[Point, Point]:
        if _bnd_ring_m is None:
            return (a, b)
        da = _bnd_ring_m.distance(ShapelyPoint(a[0], a[1]))
        db = _bnd_ring_m.distance(ShapelyPoint(b[0], b[1]))
        anchor, free = (a, b) if da <= db else (b, a)
        dx, dy = free[0] - anchor[0], free[1] - anchor[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return (a, b)
        ux, uy = dx / L, dy / L
        return (anchor, (anchor[0] + ux * SEPARATION_LEN_M, anchor[1] + uy * SEPARATION_LEN_M))

    separation_segments = [
        _norm_separation(_transform(s.start, ppm, page_height),
                         _transform(s.end, ppm, page_height))
        for s in vectors.separation
    ]

    sub_plot_labels = [
        SubPlotLabel(
            label=det.text,
            position=_transform(det.center, ppm, page_height),
        )
        for det in anchor_result.sub_plot_detections
    ]

    # Parenthesised chain measurements routed directly (no anchor line available
    # since chain candidates are excluded from anchoring to avoid wrong snaps).
    # Position: snap to the nearest chain line at the same uniform offset so the
    # chain dims sit off their line like every other label.  Rotation still uses
    # the glyph's OWN PCA orientation (det.angle_deg), unambiguous where chain
    # lines cross; the DXF y-flip + (-90,90] normalisation keeps it upright.
    chain_dim_measurements = [
        Measurement(
            raw=det.text,
            value=None,
            source=MeasurementSource.OCR,
            confidence=det.confidence,
            line_ref=None,
            line_class="chain",
            position=_snap_to_nearest_chain(
                _transform(det.center, ppm, page_height),
                chain_segments,
                label_offset,
            ),
            line_length_m=None,
            line_angle=_glyph_angle_to_dxf(det.angle_deg),
        )
        for det in anchor_result.chain_dim_detections
    ]
    measurements = measurements + chain_dim_measurements

    # Separate any labels that genuinely overlap (dense centre), sliding each
    # along its own line so it stays at the uniform offset.  Reuses label_height
    # (matches to_dxf.text_height_for) for accurate bbox sizes.
    _resolve_label_overlaps(measurements, label_height, min_gap_m=1.0)

    neighbor_labels = [
        NeighborLabel(
            label=det.text,
            position=_transform(det.center, ppm, page_height),
        )
        for det in anchor_result.neighbor_labels
    ]

    dashed_ref_segments = [
        (_transform(s.start, ppm, page_height), _transform(s.end, ppm, page_height))
        for s in vectors.dashed_ref
    ]

    # Bug 4: find the survey-number glyph sentinel injected by the glyph pass.
    survey_glyph_center: Point | None = None
    for det in detections:
        if det.kind == "survey_number_glyph":
            survey_glyph_center = _transform(det.center, ppm, page_height)
            break

    return Plot(
        client_id=client_id,
        survey_no=header.survey_no or "",
        district=header.district or "",
        taluk=header.taluk or "",
        village=header.village or "",
        stated_area=header.stated_area_ha,
        scale=header.scale_denominator,
        boundary=boundary,
        corner_points=corner_points,
        measurements=measurements,
        sub_plot_labels=sub_plot_labels,
        neighbor_labels=neighbor_labels,
        dashed_ref_segments=dashed_ref_segments,
        survey_glyph_center=survey_glyph_center,
        subdivision_segments=subdivision_segments,
        chain_segments=chain_segments,
        separation_segments=separation_segments,
        status=PlotStatus.EXTRACTED,
    )
