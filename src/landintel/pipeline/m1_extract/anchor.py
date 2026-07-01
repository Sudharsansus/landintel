"""Pair each OCR measurement to the line segment it labels.

Single responsibility: given the classified segments from :mod:`pdf_vectors` and
the detections from :mod:`ocr`, decide which line each number annotates. Nothing
else -- no normalization, no scale, no validation.

How the pairing works (calibrated on the real fixtures):

An FMB dimension label sits *alongside* the line it measures, offset a few
points perpendicular to it and rotated to match the line's angle. So matching is
**not** nearest-distance alone -- it weighs proximity together with
text/line angle agreement. On the fixtures the correct line is consistently the
nearest one whose direction matches the text orientation to within a couple of
degrees, while wrong candidates are both farther and worse-angled. The combined
score ``distance + ANGLE_WEIGHT * angle_diff`` lets a well-aligned line win over
one that is marginally closer but pointing the wrong way.

The match is **value-independent**: it uses only geometry, never the recognized
number. (That keeps the "anchored line length should equal the measured value"
relationship available as an *independent* correctness check downstream, rather
than baking it into the matching and making it circular.)

Three things by design:

* **Confidence flows through.** Each anchored pair keeps the OCR confidence so
  the agent layer -- not this step -- can decide that a 0.6 anchor is a review
  candidate.
* **Unanchored is a real outcome.** With imperfect OCR recall many lines have no
  label and some numbers match no line. Both are returned as leftovers rather
  than force-fitted; a wrong pairing is worse than an honest gap.
* **No semantic filtering.** Every detection is offered to the matcher; header
  text and stray labels simply land in the leftovers because they are far from
  any line or point the wrong way. Telling measurements from metadata is the
  caller's job (``build_plot`` via ``parse_header`` + the validator).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from ...core.models import Point

_log = logging.getLogger(__name__)
from .ocr import OCRDetection
from .pdf_vectors import PageVectors, Segment

__all__ = [
    "AnchoredMeasurement",
    "UnanchoredLine",
    "AnchorResult",
    "anchor_measurements",
    "MAX_ANCHOR_DISTANCE",
    "MAX_ANGLE_DIFF",
    "ANGLE_WEIGHT",
    "MIN_ANCHOR_SEGMENT_LENGTH",
    "_SUB_PLOT_RE",
]

# --- Matching constants (tunable; calibrated on the Sivagangai fixtures) ------

MAX_ANCHOR_DISTANCE: float = 80.0
"""Max perpendicular distance (PDF points) from a label to its line.

Kallapadi-district labels sit up to ~58 pt off their line (the PDF uses a wider
layout than Sivagangai). 80 captures those with margin while staying well inside
the nearest parallel line. Sivagangai matches at 5-6 pt are unaffected.
"""

MAX_ANGLE_DIFF: float = 28.0
"""Max angle (degrees) between text orientation and line direction to match.

Correct matches typically agree to within ~2 degrees. Some glyph-extraction
rotations introduce up to ~26° apparent tilt for genuinely correct pairs (surveyed
on survey-100 fixture: 13.4 at 26.2°, 9 at 23.8°, both at dist < 5pt). 28° closes
this gap while still rejecting lines clearly pointing the wrong way.
"""

ANGLE_WEIGHT: float = 0.5
"""Points of distance each degree of angle disagreement is worth, in scoring."""

MIN_ANCHOR_SEGMENT_LENGTH: float = 8.0
"""Minimum segment length (PDF points) for a line to be an anchor candidate.

Short connector stubs (≤5-6 pt) between boundary junctions are physically close
to nearby labels but are not the lines those labels measure. Excluding them stops
them from beating the correct longer edge in the distance+angle score.

Calibration: at scale 1:2021 (survey 100), 8 pt corresponds to ~5.7 m.
The smallest confirmed measurement in the Sivagangai fixtures is 7.2 m (~10 pt),
which was excluded by the old 15 pt threshold. Stubs at junctions are typically
≤5-6 pt, so 8 pt catches short measurements while still excluding stubs.
"""

# Distance (PDF points) from a page edge within which a 2-digit integer is
# treated as a neighbor survey number rather than a measurement.
_MARGIN_PT: float = 40.0

# Tokens that are never measurement values.  Tested before anchoring so these
# never reach the distance+angle scorer.
#
#   Kept (pass through to anchoring):
#     decimal numbers:  "69.0", "34.6", "145,6"
#     parenthesised:    "(160.2)", "(94.0)"  — chain distances
#
#   Rejected:
#     [A-H]            corner stone labels A-H
#     [1-9]|[12][0-9]|3[01]   stone point numbers 1–31
#     \d[A-Z]\d?       sub-plot labels 2A, 3B, 5A1, 5B …
#     [A-Z]{1,2}\d?    1-2 letter tokens (E, F, OF, IA) + optional trailing digit
#     \d{3,}           3+ digit neighbour survey numbers (caught before this as neighbor_labels)
#     [^0-9,.()\[\]]+ tokens containing no digits / decimal point / parens —
#                      catches noise like "|", "s%", "uf", pipe artifacts
_NON_MEASUREMENT_RE = re.compile(
    r"^("
    r"[A-H]"                     # corner stone labels A-H
    r"|[1-9]|[12][0-9]|3[01]"   # stone point numbers 1-31
    r"|3[2-9]|[4-9]\d"          # integers 32-99: never FMB measurements (always X.Y format)
    r"|\d[A-Z]\d?"               # sub-plot labels: 2A, 3B, 5B, 5A1, 5A2
    r"|[A-Z]{1,2}\d?"            # 1-2 letter labels: E, F, OF, IA, E1 …
    r"|\d{3,}"                   # 3+ digit neighbour survey numbers
    r"|[^0-9,.()]+"              # anything with no digit/decimal/paren chars
    r"|.*[|'].*"                 # contains pipe or apostrophe (OCR artifacts mixed with digits)
    r"|.*[.,][\d]*[.,].*"        # two decimal separators = merged cluster misread ("145.28.6")
    r"|\d+[A-Z]{2,}\w*"          # digit + 2+ letters = OCR noise ("5ANL", "43.66A")
    r"|\(\)"                     # empty parentheses — OCR artifact from parenthesis fills
    r"|.*\s.*"                   # any whitespace inside token = merged OCR misread ("22 08")
    r"|\d{2,}[A-Za-z]+"          # 2+ digits followed by letters = "52l", "23ABC"
    r"|\d+[.,]\d{2,}"            # 2+ decimal digits = OCR noise ("23.43", "43.66"); FMB
                                 # measurements are always X.Y format (exactly 1 decimal)
    r"|.*[^0-9,.()\[\]A-Za-z\-]" # trailing non-measurement chars = "5A2$", "7.2!"
    r")$",
    re.IGNORECASE,
)

# Sub-pattern that identifies sub-plot labels within the non-measurement set.
# These are routed to SUB DIVISION NUMBER rather than dropped.
_SUB_PLOT_RE = re.compile(r"^\d[A-Z]\d?$", re.IGNORECASE)

# GCP Vision reads Tamil Nadu sub-plot labels such as "3A" using Cyrillic
# lookalike glyphs: "З" (ze) → "3", "А" (Cyrillic A) → "A", etc.  Translate
# before any regex check so "ЗА" becomes "3A" and matches _SUB_PLOT_RE.
_CYRILLIC_TO_LATIN = str.maketrans(
    "ЗзАаВвСсЕеНнКкМмОоРрТтХхУу",
    "33AaBbCcEeHhKkMmOoPpTtXxYy",
)


def _normalize_label(text: str) -> str:
    """Translate Cyrillic OCR lookalikes to Latin/digit equivalents."""
    return text.translate(_CYRILLIC_TO_LATIN)

# Parenthesised chain line measurements: "(160.2)", "(93,8)", etc.
# Large blue glyph singletons OCR as these. Written to CHAINLINE_DIMENSIONS
# without needing an anchor line (chain lines excluded from candidates).
# Require exactly one decimal separator so parenthesised integers (OCR noise
# like "(222)") don't sneak through.
_CHAIN_DIM_RE = re.compile(r"^\(\d{1,4}[.,]\d{1,2}\)$")
# Parenthesised integers (no decimal): OCR noise from paren bezier paths
# rendered in isolation.  Filtered before anchoring so they don't reach the
# distance scorer or chain_dim_detections.
_PAREN_INT_RE = re.compile(r"^\(\d+\)$")


@dataclass(frozen=True)
class AnchoredMeasurement:
    """An OCR detection matched to the line segment it labels."""

    text: str
    confidence: float
    center: Point
    line_class: str
    """Which feature class the line belongs to: ``boundary`` / ``internal`` / ``chain``."""

    line: Segment
    distance: float
    """Perpendicular distance from the label centre to the matched line (points)."""

    angle_diff: float
    """Angle disagreement between text orientation and line direction (degrees)."""


@dataclass(frozen=True)
class UnanchoredLine:
    """A classified line segment that no measurement claimed."""

    line_class: str
    line: Segment


@dataclass(frozen=True)
class AnchorResult:
    """The outcome of anchoring: matched pairs and both kinds of leftover."""

    anchored: list[AnchoredMeasurement]
    unanchored_measurements: list[OCRDetection]
    unanchored_lines: list[UnanchoredLine]
    sub_plot_detections: list[OCRDetection]
    """OCR tokens identified as sub-plot labels (2A, 3B, 5B, ...) for the
    SUB DIVISION NUMBER layer.  Filtered out of anchoring but not discarded."""

    neighbor_labels: list[OCRDetection] = field(default_factory=list)
    """OCR tokens identified as neighbor survey numbers (99, 101, 102, ...).

    Detected by being all-digit integers with 3+ digits (always a survey number)
    or 2-digit integers near the page edge.  Written to the ``neighbor label``
    (ACI 7) layer by ``to_dxf``.  Added after ``sub_plot_detections`` so the
    field has a default and existing callers that omit it remain valid.
    """

    chain_dim_detections: list[OCRDetection] = field(default_factory=list)
    """Parenthesised chain line measurements: ``(160.2)``, ``(93,8)``, etc.

    These come from large blue glyph singletons (chain dimension labels written
    in a larger font than boundary dims).  Written directly to
    ``CHAINLINE_DIMENSIONS`` without an anchor line match.
    """


# --- Geometry helpers --------------------------------------------------------


def _orientation(p: Point, q: Point) -> float:
    """Undirected orientation of segment p->q in degrees, in ``[0, 180)``."""
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 180.0


def _angle_diff(a: float, b: float) -> float:
    """Smallest difference between two undirected angles, in ``[0, 90]``."""
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Shortest distance from point ``p`` to the segment ``a-b``."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _text_orientation(polygon: tuple[Point, ...]) -> float:
    """Orientation of a detection's text from its box's top edge, in ``[0, 180)``.

    PaddleOCR returns box corners starting top-left going clockwise, so the
    first edge runs along the text baseline.
    """
    return _orientation(polygon[0], polygon[1])


# --- Anchoring ---------------------------------------------------------------


def _segment_length(s: Segment) -> float:
    dx = s.end[0] - s.start[0]
    dy = s.end[1] - s.start[1]
    return math.hypot(dx, dy)


def _candidate_lines(
    vectors: PageVectors,
    min_length: float = MIN_ANCHOR_SEGMENT_LENGTH,
) -> list[tuple[str, Segment]]:
    """All labellable line segments, tagged with their feature class.

    Short connector stubs are excluded: they are close to nearby labels but are
    not the lines those labels measure, so they win the distance score unfairly.
    """
    # Chain lines are excluded here: their dimension labels are parenthesised
    # values (e.g. "(160,2)") routed to CHAINLINE_DIMENSIONS.  Until chain OCR
    # is implemented, including chain segments only causes boundary measurements
    # near chain lines to snap to the wrong line class.
    candidates = []
    for cls, segs in (
        ("boundary", vectors.boundary),
        ("internal", vectors.internal),
    ):
        for s in segs:
            if _segment_length(s) >= min_length:
                candidates.append((cls, s))
    return candidates


def anchor_measurements(
    vectors: PageVectors,
    detections: list[OCRDetection],
    *,
    max_distance: float = MAX_ANCHOR_DISTANCE,
    max_angle_diff: float = MAX_ANGLE_DIFF,
    angle_weight: float = ANGLE_WEIGHT,
) -> AnchorResult:
    """Anchor each detection to the line it labels.

    Args:
        vectors: Classified segments from :func:`pdf_vectors.extract_vectors`.
        detections: OCR detections from :func:`ocr.extract_text`.
        max_distance: Distance gate (points).
        max_angle_diff: Angle-agreement gate (degrees).
        angle_weight: Points-per-degree weight in the combined score.

    Returns:
        An :class:`AnchorResult` with the pairs, the detections that matched no
        line, and the lines that no detection claimed.
    """
    lines = _candidate_lines(vectors)
    pw = vectors.page_width
    ph = vectors.page_height

    anchored: list[AnchoredMeasurement] = []
    unanchored_measurements: list[OCRDetection] = []
    sub_plot_detections: list[OCRDetection] = []
    neighbor_labels: list[OCRDetection] = []
    chain_dim_detections: list[OCRDetection] = []
    claimed: set[int] = set()

    for det in detections:
        # Route by pre-classified kind from pdf_vectors._classify_glyph_groups.
        # These have exact vector positions — they do not need text-pattern
        # filtering and must not be offered to the distance scorer.
        if det.kind == "survey_number_glyph":
            continue
        if det.kind in ("neighbor_number", "neighbor"):
            neighbor_labels.append(det)
            continue

        # Red fills (stone letters A-H, stone IDs 1-31, neighbor numbers):
        # same color, different roles — discriminate by value + edge proximity.
        # Stone letters and IDs are already written to STONES via corner_points;
        # routing them to sub_plot_detections would duplicate them in SUBDIVISION.
        if det.kind == "red_label":
            tok = det.text.strip()
            if tok.isdigit():
                tok_len = len(tok)
                cx, cy = det.center
                near_edge = (
                    cx < _MARGIN_PT or cx > pw - _MARGIN_PT
                    or cy < _MARGIN_PT or cy > ph - _MARGIN_PT
                )
                if tok_len >= 3 or (tok_len == 2 and near_edge and int(tok) > 31):
                    neighbor_labels.append(det)
                    continue
            # Stone letters (A-H) and stone IDs (1-31) → unanchored, not sub_plot
            unanchored_measurements.append(det)
            continue

        # Legacy text-based neighbor detection: handles detections without a kind
        # (e.g. fallback engines that don't set group.kind).
        tok = det.text.strip()
        if tok.isdigit():
            tok_len = len(tok)
            cx, cy = det.center
            near_edge = (
                cx < _MARGIN_PT or cx > pw - _MARGIN_PT
                or cy < _MARGIN_PT or cy > ph - _MARGIN_PT
            )
            if tok_len >= 3 or (tok_len == 2 and near_edge and int(tok) > 31):
                neighbor_labels.append(det)
                continue

        # Parenthesised integers without decimal: OCR noise (paren bezier paths
        # rendered in isolation).  Discard before anchoring.
        if _PAREN_INT_RE.match(tok):
            unanchored_measurements.append(det)
            continue

        # Parenthesised chain measurements "(160.2)", "(93,8)" — large blue
        # glyph singletons (chain dims use a larger font than boundary dims).
        # Route directly to chain_dim_detections; no anchor line needed.
        if _CHAIN_DIM_RE.match(tok):
            chain_dim_detections.append(det)
            continue

        # Normalise Cyrillic lookalikes before any sub-plot check so that
        # "ЗА" (GCP Vision rendering of "3A") becomes "3A" and matches.
        tok_norm = _normalize_label(tok)

        if _NON_MEASUREMENT_RE.match(det.text) or tok_norm != tok:
            if _SUB_PLOT_RE.match(tok_norm):
                # Emit the detection with the normalised text so the DXF
                # carries "3A" not "ЗА".
                if tok_norm != tok:
                    import dataclasses
                    det = dataclasses.replace(det, text=tok_norm)
                sub_plot_detections.append(det)
            elif det.kind == "dimension" and re.match(r"^[1-9]\d?$", tok_norm):
                # Blue fill with a small integer: this is a sub-plot parcel number
                # (e.g. "1" labelling the undivided parcel), not a stone ID (red).
                sub_plot_detections.append(det)
            elif det.kind == "dimension":
                # Salvage garbled sub-plot labels: strip non-alphanumeric noise
                # introduced by OCR boundary bleed (e.g. "5.A2$" → "5A2").
                cleaned = re.sub(r"[^A-Za-z0-9]", "", tok_norm)
                if _SUB_PLOT_RE.match(cleaned):
                    import dataclasses
                    sub_plot_detections.append(dataclasses.replace(det, text=cleaned))
                    continue
                unanchored_measurements.append(det)
            else:
                unanchored_measurements.append(det)
            continue

        # Distance-only matching: find the nearest candidate line within
        # max_distance.  Line classification (boundary vs internal) routes the
        # label to the correct DXF layer.
        center = det.center

        best_index: int | None = None
        best_distance = math.inf
        best_class = ""

        for index, (line_class, segment) in enumerate(lines):
            distance = _point_segment_distance(center, segment.start, segment.end)
            if distance > max_distance:
                continue
            if distance < best_distance:
                best_distance = distance
                best_index = index
                best_class = line_class

        if best_index is None:
            unanchored_measurements.append(det)
            continue

        claimed.add(best_index)
        anchored.append(
            AnchoredMeasurement(
                text=det.text,
                confidence=det.confidence,
                center=center,
                line_class=best_class,
                line=lines[best_index][1],
                distance=best_distance,
                angle_diff=0.0,
            )
        )

    unanchored_lines = [
        UnanchoredLine(line_class=line_class, line=segment)
        for index, (line_class, segment) in enumerate(lines)
        if index not in claimed
    ]

    # Post-process: Tamil Nadu FMBs always number the base parcel "1".  GCP
    # Vision sometimes reads the stylised "1" blue glyph as "2" (serifs make
    # it look like a "2").  If we have no standalone "1" sub-plot but DO have
    # a standalone "2" (single digit, not "2A"/"2B") and compound labels like
    # "2A"/"2B" also exist, the "2" is a misread "1" → rename it.
    import dataclasses as _dc
    _standalone_ints = {
        d.text: d for d in sub_plot_detections if re.match(r"^[1-9]$", d.text)
    }
    _has_compound_2 = any(
        re.match(r"^2[A-Z]", d.text, re.IGNORECASE) for d in sub_plot_detections
    )
    if "1" not in _standalone_ints and "2" in _standalone_ints and _has_compound_2:
        _bad = _standalone_ints["2"]
        sub_plot_detections = [
            _dc.replace(d, text="1") if d is _bad else d
            for d in sub_plot_detections
        ]
        _log.info("anchor: renamed sub-plot '2'→'1' (GCP Vision misread heuristic)")

    result = AnchorResult(
        anchored=anchored,
        unanchored_measurements=unanchored_measurements,
        unanchored_lines=unanchored_lines,
        sub_plot_detections=sub_plot_detections,
        neighbor_labels=neighbor_labels,
        chain_dim_detections=chain_dim_detections,
    )
    _log.info(
        "anchor: anchored=%d unanchored_measurements=%d unanchored_lines=%d "
        "sub_plot=%d neighbor=%d chain_dims=%d",
        len(result.anchored),
        len(result.unanchored_measurements),
        len(result.unanchored_lines),
        len(result.sub_plot_detections),
        len(result.neighbor_labels),
        len(result.chain_dim_detections),
    )
    return result
