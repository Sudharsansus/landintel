"""Per-element glyph extraction and normalization for FMB dimension numbers.

FMB dimension numbers are encoded as small filled vector paths in the PDF — each
digit is a filled glyph path. Grouping nearby paths by proximity gives one cluster
per measurement number. Rendering each cluster to a clean isolated canvas (scaled
to ~70px height, de-rotated to horizontal) turns a rotated speck on a busy page
into a big clean horizontal number on a white card — the step that takes OCR recall
from ~54% toward ~90%+.

The technique mirrors what competitors call 'MakeSvgImage': isolate each number,
normalise geometry, de-skew, then OCR individually.

Color encoding in Tamil Nadu FMBs (verified on 46 Sivagangai fixtures):
  Blue (0,0,1) fills  -> dimension measurement labels (e.g. 69.0, 145.6)
  Red  (1,0,0) fills  -> stone labels (A/B/C) and neighbor survey numbers
  Black (0,0,0) fills -> some boundary measurement labels

The previous color filter was near-black only — it discarded 73 of 78 fills per
page (all blue and red). This module now captures all three.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import fitz
import numpy as np
from scipy.spatial import cKDTree

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodingDiagnostic:
    """Vector-vs-raster breakdown for one FMB page.

    Answers the key question: should dimension numbers be extracted via glyph
    paths (vector -> 100%, no OCR) or via image OCR (raster -> Vision/Paddle)?

    Attributes:
        vector_glyph_groups: total clusters of small filled vector paths on the
            page — each cluster is one potential vector-encoded number.
        vector_glyph_blue: clusters that are blue-filled (dimension measurements).
        vector_glyph_red: clusters that are red-filled (stone labels / neighbors).
        vector_glyph_black: clusters that are near-black-filled (boundary dims).
        raster_image_regions: distinct image XObjects on the page.
        native_text_chars: characters in the PDF's text stream. Near-zero for all
            Tamil Nadu FMBs — the generator never writes numbers as native text.
    """

    vector_glyph_groups: int
    vector_glyph_blue: int
    vector_glyph_red: int
    vector_glyph_black: int
    raster_image_regions: int
    native_text_chars: int

    @property
    def glyph_numbers_only(self) -> bool:
        """True when the PDF's raster images are logos/seals, not dimension numbers.

        Verified on 46 Sivagangai fixtures: all have exactly 3 raster image XObjects
        that are small logos/seals (54x54, 131x132, 484x339 px placed in the header).
        The dimension numbers are colored vector fills in the drawing body.
        """
        return self.raster_image_regions <= 5  # logos only (3 typical)

    def summary(self) -> str:
        if self.glyph_numbers_only:
            note = (
                f"{self.vector_glyph_groups} vector fills "
                f"(blue={self.vector_glyph_blue} dims, "
                f"red={self.vector_glyph_red} labels, "
                f"black={self.vector_glyph_black}); "
                f"{self.raster_image_regions} raster XObjects are logos/seals only"
            )
        else:
            note = (
                f"{self.vector_glyph_groups} vector glyph groups; "
                f"{self.raster_image_regions} raster image regions "
                f"(may contain dimension numbers -> use Vision OCR)"
            )
        return f"native_text_chars={self.native_text_chars} | {note}"


__all__ = [
    "GlyphGroup",
    "EncodingDiagnostic",
    "extract_glyph_groups",
    "diagnose_encoding",
    "render_glyph_group",
    "render_member_glyph",
    "member_pca_order",
    "split_glyph_group_by_gap",
    "split_glyph_group_by_gap_2d",
    "canvas_to_pdf",
    "glyph_polygon",
    "largest_blue_glyph",
    "path_to_contour",
    "compute_hu",
    "build_digit_templates",
    "recognize_digit_by_shape",
    "fix_6_9",
    "CANVAS_W",
    "CANVAS_H",
    "CHAR_CANVAS_W",
    "CHAR_CANVAS_H",
]

CANVAS_W: int = 520
CANVAS_H: int = 150
_GLYPH_H: int = 70  # target text height in canvas units

# Per-character canvas: square, centred on a single digit glyph.
CHAR_CANVAS_W: int = 200
CHAR_CANVAS_H: int = 200
_CHAR_GLYPH_H: int = 120  # target height for a single character render

# Tuning constants — calibrated for the Sivagangai fixtures.
_MAX_GLYPH_DIM: float = 25.0       # pt; larger fills are stones/separators, not digit glyphs
_CLUSTER_GAP: float = 15.0         # pt; gap for black fills (boundary dims)
_CLUSTER_GAP_RED: float = 12.0     # pt; red fills — wide enough for 2-digit stone IDs
                                    # (~8-10pt between digit fills) but tighter than 15
                                    # to avoid merging fills from different stones
_CLUSTER_GAP_BLUE: float = 5.0     # pt; tighter gap for blue fills to separate adjacent
                                    # measurement texts (chain arrows are 10-40pt apart;
                                    # digit fills within one number are 1-4pt apart)
_MIN_AREA: float = 3.0             # pt²; smaller groups are speck noise

# Channel dominance threshold for color classification (matching pdf_vectors.py).
_COLOR_DOMINANCE: float = 0.4


@dataclass
class GlyphGroup:
    """One measurement number: a proximity cluster of filled glyph paths.

    The ``drawings`` list holds the raw fitz drawing dicts whose vector paths
    are the digit outlines. ``center`` is the anchor position in PDF page space.
    ``angle_deg`` encodes the text baseline direction using the same
    ``atan2(dy, dx) % 180`` convention as ``anchor._orientation``, so the
    rotated-rectangle polygon produced by ``glyph_polygon`` gives the correct
    angle difference for anchoring.

    ``color`` is the fill color class: ``"blue"`` for dimension measurements,
    ``"red"`` for stone labels / neighbor survey numbers, ``"black"`` for
    near-black boundary measurement labels. Defaults to ``"black"`` so existing
    code that constructs ``GlyphGroup`` without this field continues to work.

    ``kind`` is set by the vector-first classification pass in
    ``pdf_vectors.extract_vectors()``.  Blue groups are split by bbox area:
    the largest is ``"survey_number"``, large ones are ``"parcel_label"``, small
    ones are ``"dimension"``.  Red groups are ``"red_label"`` (stone letters +
    neighbor survey numbers — the value tells them apart after OCR).  Black
    groups are ``"neighbor"`` when near a page edge, otherwise ``"black_label"``.
    """

    drawings: list[dict]
    bbox: fitz.Rect
    center: tuple[float, float]
    angle_deg: float  # [0, 180) — 0 = horizontal, same as anchor._orientation
    color: str = "black"  # "blue" | "red" | "black"
    kind: str = "unknown"
    """Semantic role: "dimension" | "parcel_label" | "survey_number" |
    "red_label" | "neighbor" | "black_label" | "unknown"."""


def _fill_color_class(r: float, g: float, b: float) -> str | None:
    """Return fill color class for a text-glyph fill, or None to skip.

    Accepts the three text-fill colors found in Tamil Nadu FMBs:
      - Near-black: boundary/internal measurement glyphs
      - Blue  (0,0,1): dimension measurement labels
      - Red   (1,0,0): stone labels (A/B/C) and neighbor survey numbers
    Rejects white/near-white backgrounds and any other color (yellow chain
    markers, green fills, etc.) that are not measurement text.
    """
    if r > 0.85 and g > 0.85 and b > 0.85:
        return None  # white/near-white background
    if r <= 0.15 and g <= 0.15 and b <= 0.15:
        return "black"
    if b > 0.5 and (b - r) > _COLOR_DOMINANCE and (b - g) > _COLOR_DOMINANCE:
        return "blue"
    if r > 0.5 and (r - g) > _COLOR_DOMINANCE and (r - b) > _COLOR_DOMINANCE:
        return "red"
    return None  # other colors — not a text glyph fill


def extract_glyph_groups(page: fitz.Page) -> list[GlyphGroup]:
    """Cluster small filled text-color paths into per-number glyph groups.

    Captures near-black (boundary dims), blue (dimension measurements), and red
    (stone labels / neighbor survey numbers) fills. Clusters are color-homogeneous
    — blue and red fills never merge into the same group, ensuring each rendered
    canvas contains exactly one semantic label type.

    Previously only near-black fills were captured, discarding all blue and red
    measurements (73 of 78 per page on the Sivagangai fixtures). This version
    captures all three classes for ~100% recall before rendering.
    """
    drawings = page.get_drawings()

    # candidates: list of (drawing_dict, color_class_str)
    candidates: list[tuple[dict, str]] = []
    for d in drawings:
        fill = d.get("fill")
        if fill is None:
            continue  # stroke-only — line geometry, not a glyph fill
        r, g, b = fill[:3]
        color = _fill_color_class(r, g, b)
        if color is None:
            continue  # white background, yellow marker, or other non-text color
        rect = d["rect"]
        if rect.is_empty or rect.is_infinite:
            continue
        if rect.width > _MAX_GLYPH_DIM or rect.height > _MAX_GLYPH_DIM:
            continue  # too large to be a digit (corner stone, scale bar, separator)
        # Red text glyphs include TWO-DIGIT stone numbers (20, 21, … 31), which
        # render as a single compound fill ~9pt wide × 6pt tall (w/h ≈ 1.5).
        # A genuine stone *marker* bar is much flatter (w/h > 2.3) — keep the
        # two-digit numbers, drop only the flat marker bars.  (Dropping at the
        # old 1.3 threshold silently discarded every two-digit stone label,
        # leaving ~half the stones unlabelled vs the client reference.)
        if color == "red" and rect.width > rect.height * 2.3:
            continue
        candidates.append((d, color))

    if not candidates:
        return []

    # ── Point-to-point single-linkage clustering, per color ──────────────────
    # Splitting by color first ensures blue/red/black fills never merge across
    # color classes, which would corrupt the semantic separation of dimension
    # labels, stone labels, and boundary measurement labels.
    by_color: dict[str, list[tuple[dict, str]]] = {}
    for item in candidates:
        by_color.setdefault(item[1], []).append(item)

    _GAP_BY_COLOR: dict[str, float] = {
        "blue":  _CLUSTER_GAP_BLUE,
        "red":   _CLUSTER_GAP_RED,
        "black": _CLUSTER_GAP,
    }

    groups: list[GlyphGroup] = []
    for color, color_cands in by_color.items():
        gap = _GAP_BY_COLOR.get(color, _CLUSTER_GAP)
        for comp in _cluster_by_points(color_cands, gap):
            cluster = [color_cands[k][0] for k in comp]
            bbox = _union_rects([d["rect"] for d in cluster])
            if bbox.get_area() < _MIN_AREA:
                continue
            cx = (bbox.x0 + bbox.x1) / 2
            cy = (bbox.y0 + bbox.y1) / 2
            angle = _compute_angle_pca(cluster)
            groups.append(GlyphGroup(cluster, bbox, (cx, cy), angle, color=color))

    # Second pass: large BLUE fills (> _MAX_GLYPH_DIM in one dimension) that were
    # filtered above are sub-plot label compound paths (e.g. "1", "2A", "3B").
    # Add each as a singleton IF its center is not already near an existing cluster
    # (distance guard = 30 pt) so it never disrupts dimension measurement clusters.
    _MAX_SUBPLOT_DIM: float = 40.0
    _SINGLETON_GUARD_PT: float = 30.0
    for d in drawings:
        fill = d.get("fill")
        if fill is None:
            continue
        r, g, b = fill[:3]
        if not (b - max(r, g) > _COLOR_DOMINANCE):
            continue  # non-blue fill
        rect = d["rect"]
        if rect.is_empty or rect.is_infinite:
            continue
        # Must exceed normal threshold (already handled above) but be ≤ subplot cap
        if not (rect.width > _MAX_GLYPH_DIM or rect.height > _MAX_GLYPH_DIM):
            continue
        if rect.width > _MAX_SUBPLOT_DIM or rect.height > _MAX_SUBPLOT_DIM:
            continue  # genuinely too large (frame border, etc.)
        if rect.get_area() < _MIN_AREA:
            continue
        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2
        # Only add if no existing LARGE cluster (both dimensions > _MAX_GLYPH_DIM)
        # is within the guard radius.  Small red/black/blue clusters (stones,
        # dimension dots, arrows) sit near sub-plot fills and must NOT block them.
        too_close = False
        guard_r2 = _SINGLETON_GUARD_PT * _SINGLETON_GUARD_PT
        for g in groups:
            # Skip any cluster where neither dimension qualifies as "large".
            if not (g.bbox.width > _MAX_GLYPH_DIM and g.bbox.height > _MAX_GLYPH_DIM):
                continue
            dx = cx - g.center[0]
            dy = cy - g.center[1]
            if (dx * dx + dy * dy) <= guard_r2:
                too_close = True
                break

        if too_close:
            continue

        # Compound chain-dimension fills carry their own rotation in the path
        # outline — recover it so the label can be rendered along its line.
        singleton_angle = _compute_singleton_angle_pca(d)
        groups.append(GlyphGroup([d], rect, (cx, cy), angle_deg=singleton_angle, color="blue"))

    counts: dict[str, int] = {"blue": 0, "red": 0, "black": 0}
    for g in groups:
        counts[g.color] = counts.get(g.color, 0) + 1
    _log.info(
        "glyph groups: total=%d blue=%d red=%d black=%d",
        len(groups), counts["blue"], counts["red"], counts["black"],
    )
    return groups


def diagnose_encoding(page: fitz.Page) -> EncodingDiagnostic:
    """Report vector-vs-raster encoding of number content on one FMB page.

    Fast — no OCR, no rendering. Safe to call on every page in the pipeline.
    The result is logged by ``ocr.extract_text`` so the vector/raster answer
    appears in worker logs for every PDF processed, including production uploads.
    """
    groups = extract_glyph_groups(page)
    counts: dict[str, int] = {"blue": 0, "red": 0, "black": 0}
    for g in groups:
        counts[g.color] = counts.get(g.color, 0) + 1
    raster_images = len(page.get_images(full=False))
    native_text = page.get_text("text").strip()
    return EncodingDiagnostic(
        vector_glyph_groups=len(groups),
        vector_glyph_blue=counts["blue"],
        vector_glyph_red=counts["red"],
        vector_glyph_black=counts["black"],
        raster_image_regions=raster_images,
        native_text_chars=len(native_text),
    )


def render_glyph_group(
    group: GlyphGroup,
    *,
    flip: bool = False,
    angle_override_deg: float | None = None,
) -> np.ndarray:
    """Render the glyph group onto a clean 520x150 white canvas.

    The glyphs are de-rotated to horizontal and scaled so their height is ~70px,
    centred on the canvas. Only the glyph paths themselves are reproduced — no
    surrounding line geometry from the source page — so the OCR engine sees a
    clean isolated number on a white card.

    Args:
        group: The glyph cluster to render.
        flip: When True, use the supplementary angle (180° - angle_deg) for
            de-rotation instead of -angle_deg. Measurements on lines with
            angle_deg > 90° (going visually upper-right) render reversed at
            -angle_deg; the supplementary rotation corrects this. Callers should
            try both orientations and keep the higher-confidence OCR result.
        angle_override_deg: When provided, overrides group.angle_deg for
            de-rotation. Pass the nearest vector line's angle so that diagonal
            dimension text is de-rotated using the exact line angle rather than
            the noisier glyph-shape-derived angle.

    Returns a grayscale uint8 array of shape (CANVAS_H, CANVAS_W).
    """
    bbox = group.bbox
    src_h = max(bbox.height, 1.0)
    scale = _GLYPH_H / src_h

    cx_src = (bbox.x0 + bbox.x1) / 2
    cy_src = (bbox.y0 + bbox.y1) / 2
    cx_dst = CANVAS_W / 2.0
    cy_dst = CANVAS_H / 2.0

    # De-rotation: negate the text angle to bring it to horizontal.
    # When angle_override_deg is given (from the nearest vector line), use it —
    # more reliable than the glyph-shape-derived angle for small/sparse clusters.
    # When flip=True use the supplementary angle so text going upper-right reads
    # left-to-right rather than reversed.
    base_angle = angle_override_deg if angle_override_deg is not None else group.angle_deg
    ar = math.radians(180.0 - base_angle if flip else -base_angle)
    ca, sa = math.cos(ar), math.sin(ar)

    def xform(px: float, py: float) -> fitz.Point:
        dx, dy = px - cx_src, py - cy_src
        rx = ca * dx - sa * dy
        ry = sa * dx + ca * dy
        return fitz.Point(cx_dst + rx * scale, cy_dst + ry * scale)

    tmp = fitz.open()
    tpage = tmp.new_page(width=CANVAS_W, height=CANVAS_H)
    shape = tpage.new_shape()

    for d in group.drawings:
        has_items = False
        for item in d.get("items", []):
            kind = item[0]
            if kind == "l":
                shape.draw_line(xform(item[1].x, item[1].y), xform(item[2].x, item[2].y))
                has_items = True
            elif kind == "c":
                shape.draw_bezier(
                    xform(item[1].x, item[1].y),
                    xform(item[2].x, item[2].y),
                    xform(item[3].x, item[3].y),
                    xform(item[4].x, item[4].y),
                )
                has_items = True
            elif kind == "re":
                r = item[1]
                shape.draw_quad(fitz.Quad(
                    xform(r.x0, r.y0),
                    xform(r.x1, r.y0),
                    xform(r.x0, r.y1),
                    xform(r.x1, r.y1),
                ))
                has_items = True
            elif kind == "qu":
                q = item[1]
                shape.draw_quad(fitz.Quad(
                    xform(q.ul.x, q.ul.y),
                    xform(q.ur.x, q.ur.y),
                    xform(q.ll.x, q.ll.y),
                    xform(q.lr.x, q.lr.y),
                ))
                has_items = True
        if has_items:
            shape.finish(fill=(0, 0, 0), color=None, closePath=True, even_odd=True)

    shape.commit()
    pix = tpage.get_pixmap(colorspace=fitz.csGRAY, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    tmp.close()
    return arr


def member_pca_order(group: GlyphGroup) -> list[dict]:
    """Return the group's member drawings sorted left-to-right along the PCA axis.

    Used by OCR_ENGINE=perchar to concatenate per-character OCR results in the
    correct reading order before joining them into a number string.
    """
    drawings = group.drawings
    if len(drawings) <= 1:
        return list(drawings)
    cents = np.array(
        [((d["rect"].x0 + d["rect"].x1) / 2, (d["rect"].y0 + d["rect"].y1) / 2)
         for d in drawings],
        dtype=np.float64,
    )
    X = cents - cents.mean(0)
    _, evecs = np.linalg.eigh(X.T @ X)
    axis = evecs[:, -1]
    projections = X @ axis
    order = np.argsort(projections)
    return [drawings[i] for i in order]


def split_glyph_group_by_gap(
    group: GlyphGroup,
    gap_threshold_pt: float = 5.0,
) -> list["GlyphGroup"]:
    """Split a merged GlyphGroup into sub-groups wherever members are far apart.

    Adjacent measurement labels on nearby boundary lines are often within the
    5-pt point-to-point clustering radius, so they end up in one GlyphGroup.
    The intra-number digit spacing in these FMBs is 1-3 pt; the inter-number
    spacing between neighbouring labels is 7-22 pt.  Splitting at 5 pt
    cleanly separates merged clusters without breaking single numbers.

    Returns a list with the original group unchanged when no gap exceeds the
    threshold (single-number group or no members to split).
    """
    if len(group.drawings) <= 1:
        return [group]

    ordered = member_pca_order(group)
    if len(ordered) <= 1:
        return [group]

    # Project member centers onto the PCA baseline axis.
    cents = np.array(
        [((d["rect"].x0 + d["rect"].x1) / 2, (d["rect"].y0 + d["rect"].y1) / 2)
         for d in ordered],
        dtype=np.float64,
    )
    X = cents - cents.mean(0)
    _, evecs = np.linalg.eigh(X.T @ X)
    axis = evecs[:, -1]
    projs = (X @ axis).tolist()

    # Sort by projection (already ordered by member_pca_order, but recompute
    # to match the centred projections above).
    idx_order = sorted(range(len(ordered)), key=lambda i: projs[i])
    sorted_drawings = [ordered[i] for i in idx_order]
    sorted_projs = [projs[i] for i in idx_order]

    # Collect sub-groups by splitting at large gaps.
    sub_buckets: list[list[dict]] = []
    current: list[dict] = [sorted_drawings[0]]
    for k in range(1, len(sorted_drawings)):
        gap = sorted_projs[k] - sorted_projs[k - 1]
        if gap > gap_threshold_pt:
            sub_buckets.append(current)
            current = []
        current.append(sorted_drawings[k])
    sub_buckets.append(current)

    if len(sub_buckets) == 1:
        return [group]  # No large gap — nothing to split.

    # Build a proper GlyphGroup for each sub-bucket.
    result: list[GlyphGroup] = []
    for bucket in sub_buckets:
        bbox = _union_rects([d["rect"] for d in bucket])
        cx = (bbox.x0 + bbox.x1) / 2
        cy = (bbox.y0 + bbox.y1) / 2
        angle = _compute_angle_pca(bucket)
        result.append(GlyphGroup(
            drawings=bucket,
            bbox=bbox,
            center=(cx, cy),
            angle_deg=angle,
            color=group.color,
            kind=group.kind,
        ))
    _log.debug(
        "split_glyph_group_by_gap: split %d-member group at (%.0f,%.0f) → %d sub-groups",
        len(group.drawings), group.center[0], group.center[1], len(result),
    )
    return result


def split_glyph_group_by_gap_2d(
    group: GlyphGroup,
    gap_threshold_pt: float = 5.0,
    y_gap_threshold_pt: float = 3.0,
) -> list["GlyphGroup"]:
    """2D variant: try primary PCA axis, then pure-Y axis for vertical stacks.

    Only applies to groups with >= 4 members; smaller groups are passed straight
    to the 1D splitter.  The Y-axis pass uses a tighter threshold (3 pt) because
    vertically-stacked measurements may share less separation than horizontal ones.
    """
    if len(group.drawings) < 2:
        return [group]  # Single fill — nothing to split.

    # Try primary PCA axis first.
    base = split_glyph_group_by_gap(group, gap_threshold_pt)
    if len(base) > 1:
        # 1D already split — recursively apply 2D to each sub-group so that
        # sub-groups that are themselves vertically stacked get Y-split too.
        result: list[GlyphGroup] = []
        for sg in base:
            result.extend(split_glyph_group_by_gap_2d(sg, gap_threshold_pt, y_gap_threshold_pt))
        return result

    # Primary axis failed.  Try splitting on the pure Y axis (PDF y increases
    # downward).  This catches horizontal measurements stacked vertically, e.g.
    # "(24.2)" above "20.2" — both have similar X spreads, only differ in Y.
    ordered = member_pca_order(group)
    y_centers = [(d["rect"].y0 + d["rect"].y1) / 2 for d in ordered]
    idx_order = sorted(range(len(ordered)), key=lambda i: y_centers[i])
    sorted_drw = [ordered[i] for i in idx_order]
    sorted_y = [y_centers[i] for i in idx_order]

    # Log Y-gaps to help calibrate the threshold.
    if len(sorted_y) >= 4:
        gaps = [sorted_y[k] - sorted_y[k - 1] for k in range(1, len(sorted_y))]
        _log.debug(
            "split_2d Y-gaps for %d-member group at (%.0f,%.0f): %s",
            len(group.drawings), group.center[0], group.center[1],
            " ".join(f"{g:.1f}" for g in gaps),
        )

    sub_buckets: list[list[dict]] = []
    current: list[dict] = [sorted_drw[0]]
    for k in range(1, len(sorted_drw)):
        if sorted_y[k] - sorted_y[k - 1] > y_gap_threshold_pt:
            sub_buckets.append(current)
            current = []
        current.append(sorted_drw[k])
    sub_buckets.append(current)

    if len(sub_buckets) == 1:
        return [group]

    # Absorb singletons (lone parenthesis fills).
    changed = True
    while changed and len(sub_buckets) > 1:
        changed = False
        for i, b in enumerate(sub_buckets):
            if len(b) < 2:
                if i + 1 < len(sub_buckets):
                    sub_buckets[i + 1] = b + sub_buckets[i + 1]
                else:
                    sub_buckets[i - 1] = sub_buckets[i - 1] + b
                sub_buckets.pop(i)
                changed = True
                break
    if len(sub_buckets) == 1:
        return [group]

    result: list[GlyphGroup] = []
    for bucket in sub_buckets:
        bbox = _union_rects([d["rect"] for d in bucket])
        cx = (bbox.x0 + bbox.x1) / 2
        cy = (bbox.y0 + bbox.y1) / 2
        angle = _compute_angle_pca(bucket)
        sg = GlyphGroup(
            drawings=bucket, bbox=bbox, center=(cx, cy),
            angle_deg=angle, color=group.color, kind=group.kind,
        )
        result.extend(split_glyph_group_by_gap(sg, gap_threshold_pt))

    _log.debug(
        "split_glyph_group_by_gap_2d: Y-split %d-member group at (%.0f,%.0f) → %d sub-groups",
        len(group.drawings), group.center[0], group.center[1], len(result),
    )
    return result


def render_member_glyph(
    drawing: dict,
    angle_deg: float,
    group_center: tuple[float, float],
    *,
    flip: bool = False,
) -> np.ndarray:
    """Render one member path of a glyph group as an isolated per-character canvas.

    Same transform as ``render_glyph_group`` (de-rotate by ``angle_deg``, scale
    to ``_CHAR_GLYPH_H`` px) but operates on a single glyph path and writes to
    a compact ``CHAR_CANVAS_W × CHAR_CANVAS_H`` canvas centred on that path's
    own bbox.  The result is a clean white-on-black single-digit (or single-
    symbol) image — the easiest possible OCR input.

    Args:
        drawing: One fitz drawing dict (a single filled glyph path).
        angle_deg: Group-level PCA angle used for de-rotation.  Using the whole
            group's angle is more reliable than trying to infer direction from a
            single symmetric glyph (e.g. "0", "8").
        group_center: The bbox centre of the whole group (used as the rotation
            pivot so the character stays centred after de-rotation).
        flip: When True apply the supplementary angle (180° - angle_deg).
    """
    rect = drawing["rect"]
    src_h = max(rect.height, 1.0)
    scale = _CHAR_GLYPH_H / src_h

    cx_src, cy_src = group_center
    cx_dst = CHAR_CANVAS_W / 2.0
    cy_dst = CHAR_CANVAS_H / 2.0

    base_angle = angle_deg
    ar = math.radians(180.0 - base_angle if flip else -base_angle)
    ca, sa = math.cos(ar), math.sin(ar)

    def xform(px: float, py: float) -> fitz.Point:
        dx, dy = px - cx_src, py - cy_src
        rx = ca * dx - sa * dy
        ry = sa * dx + ca * dy
        return fitz.Point(cx_dst + rx * scale, cy_dst + ry * scale)

    tmp = fitz.open()
    tpage = tmp.new_page(width=CHAR_CANVAS_W, height=CHAR_CANVAS_H)
    shape = tpage.new_shape()
    has_items = False
    for item in drawing.get("items", []):
        kind = item[0]
        if kind == "l":
            shape.draw_line(xform(item[1].x, item[1].y), xform(item[2].x, item[2].y))
            has_items = True
        elif kind == "c":
            shape.draw_bezier(
                xform(item[1].x, item[1].y),
                xform(item[2].x, item[2].y),
                xform(item[3].x, item[3].y),
                xform(item[4].x, item[4].y),
            )
            has_items = True
        elif kind == "re":
            r = item[1]
            shape.draw_quad(fitz.Quad(
                xform(r.x0, r.y0), xform(r.x1, r.y0),
                xform(r.x0, r.y1), xform(r.x1, r.y1),
            ))
            has_items = True
        elif kind == "qu":
            q = item[1]
            shape.draw_quad(fitz.Quad(
                xform(q.ul.x, q.ul.y), xform(q.ur.x, q.ur.y),
                xform(q.ll.x, q.ll.y), xform(q.lr.x, q.lr.y),
            ))
            has_items = True
    if has_items:
        shape.finish(fill=(0, 0, 0), color=None, closePath=True, even_odd=True)
    shape.commit()
    pix = tpage.get_pixmap(colorspace=fitz.csGRAY, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    tmp.close()
    return arr


def canvas_to_pdf(
    group: GlyphGroup,
    cx_canvas: float,
    cy_canvas: float,
    *,
    flip: bool = False,
    angle_override_deg: float | None = None,
) -> tuple[float, float]:
    """Map a canvas coordinate back to PDF page space.

    Inverse of the affine transform applied by render_glyph_group.  Use this
    to convert per-detection canvas centres returned by the OCR engine back to
    PDF positions so each text region in a multi-measurement cluster gets its
    own anchor position.

    angle_override_deg must match what was passed to render_glyph_group so the
    forward and inverse transforms use the same rotation matrix.
    """
    bbox = group.bbox
    src_h = max(bbox.height, 1.0)
    scale = _GLYPH_H / src_h
    cx_src = (bbox.x0 + bbox.x1) / 2
    cy_src = (bbox.y0 + bbox.y1) / 2
    cx_dst = CANVAS_W / 2.0
    cy_dst = CANVAS_H / 2.0
    base_angle = angle_override_deg if angle_override_deg is not None else group.angle_deg
    ar = math.radians(180.0 - base_angle if flip else -base_angle)
    ca, sa = math.cos(ar), math.sin(ar)
    # Undo scale and translate, then undo rotation [[ca,-sa],[sa,ca]]⁻¹ = [[ca,sa],[-sa,ca]]
    dx = (cx_canvas - cx_dst) / scale
    dy = (cy_canvas - cy_dst) / scale
    px = cx_src + ca * dx + sa * dy
    py = cy_src + (-sa) * dx + ca * dy
    return px, py


def largest_blue_glyph(groups: list[GlyphGroup]) -> GlyphGroup | None:
    """Return the blue glyph group with the largest bounding-box area, or None.

    The survey number (e.g. "100") is drawn as a large blue filled-glyph label
    at the plot centroid.  It is the biggest blue cluster on the page — finding
    it by max bbox area is a robust heuristic across all Sivagangai fixtures.

    Used by the glyph-extraction path to inject a survey-glyph-center sentinel
    into the OCR detection list so ``build_plot`` can place the survey-number
    label at the drawn position rather than the geometric centroid.
    """
    blue = [g for g in groups if g.color == "blue"]
    if not blue:
        return None
    return max(blue, key=lambda g: g.bbox.get_area())


def glyph_polygon(group: GlyphGroup) -> tuple[tuple[float, float], ...]:
    """Rotated bounding rectangle in PDF page space encoding the text angle.

    Returns four corners clockwise [top-left, top-right, bottom-right, bottom-left]
    such that polygon[0]->polygon[1] has direction ``angle_deg`` -- matching the
    ``anchor._orientation`` convention so the anchoring angle-match works correctly
    for rotated dimension labels.
    """
    cx, cy = group.center
    hw = group.bbox.width / 2
    hh = group.bbox.height / 2
    ar = math.radians(group.angle_deg)
    ca, sa = math.cos(ar), math.sin(ar)

    def rot(ldx: float, ldy: float) -> tuple[float, float]:
        return (cx + ca * ldx - sa * ldy, cy + sa * ldx + ca * ldy)

    # Top edge (polygon[0]->polygon[1]) direction = (ca, sa) = angle_deg. ✓
    return (
        rot(-hw, -hh),  # top-left
        rot(hw, -hh),   # top-right
        rot(hw, hh),    # bottom-right
        rot(-hw, hh),   # bottom-left
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_contour_pts(d: dict) -> list[tuple[float, float]]:
    """Actual anchor/control points of a fitz drawing dict.

    Used instead of the axis-aligned bounding rect so that the clustering
    distance test operates on true geometry rather than AABB extents.  AABB
    gaps grow with rotation angle (inflated by cos θ + sin θ), making them
    unreliable for steeply-rotated dimension text.  Point-to-point distance
    is rotation-invariant.

    Handles line (`l`), Bézier (`c`), rectangle (`re`), and quad (`qu`) items.
    Move (`m`) and close (`h`) carry no new geometry and are skipped.
    """
    pts: list[tuple[float, float]] = []
    for item in d.get("items", []):
        tag = item[0]
        if tag == "l":
            pts.append((item[1].x, item[1].y))
            pts.append((item[2].x, item[2].y))
        elif tag == "c":
            for k in (1, 2, 3, 4):
                pts.append((item[k].x, item[k].y))
        elif tag == "re":
            r = item[1]
            pts += [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
        elif tag == "qu":
            q = item[1]
            pts += [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y),
                    (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
    return pts


def _cluster_by_points(
    candidates: list[tuple[dict, str]],
    gap: float,
) -> list[list[int]]:
    """Single-linkage clustering on actual path anchor points.

    Replaces the AABB-gap approach (``_rects_close``) which fails for steeply
    rotated text.  Two glyph paths are connected when any pair of their anchor
    points is within ``gap`` PDF points — a rotation-invariant test.  Uses a
    KD-tree for O(n log n) pair queries and union-find for component tracking.

    All inputs must share the same color class; cross-color merges are handled
    by calling this function separately per color.
    """
    n = len(candidates)
    if n == 0:
        return []

    all_pts: list[tuple[float, float]] = []
    owner: list[int] = []
    for i, (d, _) in enumerate(candidates):
        pts = _extract_contour_pts(d)
        if not pts:
            r = d["rect"]
            pts = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
        for pt in pts:
            all_pts.append(pt)
            owner.append(i)

    pts_arr = np.array(all_pts, dtype=np.float32)
    owner_arr = np.array(owner, dtype=np.int32)

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    tree = cKDTree(pts_arr)
    for ai, bi in tree.query_pairs(r=gap):
        oa, ob = int(owner_arr[ai]), int(owner_arr[bi])
        if oa != ob:
            union(oa, ob)

    comp: dict[int, list[int]] = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)
    return list(comp.values())


def _compute_angle_pca(drawings: list[dict]) -> float:
    """Dominant text-baseline angle in [0, 180) from PCA of glyph centres.

    PCA finds the principal axis of the centroid scatter, which is the
    baseline direction of the number string regardless of rotation.
    For single-glyph groups (e.g. a chain-dimension compound fill that holds
    the whole "(160.2)" as one path) the centroid scatter is a single point,
    so fall back to PCA over that path's own outline points.
    """
    if len(drawings) < 2:
        return _compute_singleton_angle_pca(drawings[0]) if drawings else 0.0
    cents = np.array(
        [((d["rect"].x0 + d["rect"].x1) / 2, (d["rect"].y0 + d["rect"].y1) / 2)
         for d in drawings],
        dtype=np.float64,
    )
    X = cents - cents.mean(0)
    _, evecs = np.linalg.eigh(X.T @ X)
    axis = evecs[:, -1]  # eigenvector of largest eigenvalue
    return math.degrees(math.atan2(axis[1], axis[0])) % 180.0


def _compute_singleton_angle_pca(drawing: dict) -> float:
    """Text-baseline angle in [0, 180) of one compound glyph from its outline.

    A chain-dimension label like "(160.2)" is encoded as a single rotated
    compound fill path.  Its centroid carries no direction, but the scatter of
    the path's own points does: the principal axis of the outline is the text
    baseline.  Returns 0.0 when the path is too sparse to be directional.
    """
    contour = path_to_contour(drawing, bezier_samples=6)
    if contour is None:
        return 0.0
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return 0.0
    X = pts - pts.mean(0)
    evals, evecs = np.linalg.eigh(X.T @ X)
    # Need a clearly elongated scatter; a near-square blob has no baseline.
    if evals[-1] < 1e-6 or evals[-1] < 1.3 * evals[0]:
        return 0.0
    axis = evecs[:, -1]
    return math.degrees(math.atan2(axis[1], axis[0])) % 180.0


def _rects_close(a: fitz.Rect, b: fitz.Rect, gap: float) -> bool:
    """True if two rects are within ``gap`` PDF points of each other."""
    expanded = a + (-gap, -gap, gap, gap)
    return not expanded.intersect(b).is_empty


def _union_rects(rects: list[fitz.Rect]) -> fitz.Rect:
    result = rects[0]
    for r in rects[1:]:
        result |= r
    return result


def _compute_angle(rects: list[fitz.Rect]) -> float:
    """Dominant text angle in [0, 180) from glyph-centre arrangement.

    Uses ``atan2(dy, dx) % 180`` in PDF space (y increases downward), matching
    ``anchor._orientation``. Falls back to 0.0 for single-glyph groups where
    direction is undefined.
    """
    if len(rects) < 2:
        return 0.0
    centers = sorted(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2) for r in rects)
    dx = centers[-1][0] - centers[0][0]
    dy = centers[-1][1] - centers[0][1]
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return 0.0
    return math.degrees(math.atan2(dy, dx)) % 180.0


# ---------------------------------------------------------------------------
# Vector shape-matching helpers (OCR_ENGINE=shapematch)
# ---------------------------------------------------------------------------


def path_to_contour(drawing: dict, bezier_samples: int = 8) -> np.ndarray | None:
    """Convert a fitz drawing dict to an OpenCV-style contour (N,1,2) float32.

    Samples bezier curves at ``bezier_samples`` points each so curved digit
    outlines contribute enough points for accurate moment computation. Returns
    None when the path has fewer than 5 points (too degenerate for moments).
    """
    pts: list[tuple[float, float]] = []
    for item in drawing.get("items", []):
        tag = item[0]
        if tag == "l":
            pts.append((item[1].x, item[1].y))
            pts.append((item[2].x, item[2].y))
        elif tag == "c":
            p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
            for u in np.linspace(0.0, 1.0, bezier_samples):
                v = 1.0 - u
                x = v**3 * p0.x + 3*v**2*u * p1.x + 3*v*u**2 * p2.x + u**3 * p3.x
                y = v**3 * p0.y + 3*v**2*u * p1.y + 3*v*u**2 * p2.y + u**3 * p3.y
                pts.append((x, y))
        elif tag == "re":
            r = item[1]
            pts += [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
        elif tag == "qu":
            q = item[1]
            pts += [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y),
                    (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]
    if len(pts) < 5:
        return None
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def compute_hu(contour: np.ndarray) -> np.ndarray:
    """7 log-transformed Hu moments: rotation, scale, translation invariant.

    Log transform ``-sign(h) * log10(|h| + ε)`` brings all 7 moments into a
    comparable numerical range so L2 distance in this space is meaningful.
    """
    import cv2  # noqa: PLC0415
    M = cv2.moments(contour)
    hu = cv2.HuMoments(M).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


# Template type: char → list of (hu_array, contour) so we can use either
# L2 distance in Hu space or cv2.matchShapes on the contour directly.
DigitTemplates = dict[str, list[tuple[np.ndarray, np.ndarray]]]


def build_digit_templates(
    groups: list[GlyphGroup],
    known: list[tuple[str, tuple[float, float]]],
) -> DigitTemplates:
    """Build per-digit Hu-moment templates from paddle-confirmed measurements.

    Args:
        groups: all glyph groups extracted from the page (already classified).
        known: list of ``(measurement_text, pdf_center)`` from paddle detections
               with high confidence, e.g. ``[("69.0", (152.3, 386.1)), ...]``.

    For each known measurement we find the nearest dimension glyph group, verify
    its member-path count matches the character count of the measurement string,
    sort the paths in PCA reading order, and label each path with its character.
    Those labeled paths become the template library used by
    :func:`recognize_digit_by_shape`.
    """
    templates: DigitTemplates = {}

    # Index dimension groups for fast nearest-neighbor lookup.
    dim_groups = [g for g in groups if g.kind == "dimension"]
    if not dim_groups:
        return templates

    # Characters we build digit templates for — parentheses are punctuation
    # around the whole number, not individual glyph paths.
    _DIGIT_CHARS = set("0123456789.")

    for text, (cx, cy) in known:
        # Strip parentheses to get the digit-only character sequence.
        chars = [c for c in text if c in _DIGIT_CHARS]
        if not chars:
            continue

        # Find nearest dimension group.
        best_g, best_d = None, float("inf")
        for g in dim_groups:
            gx, gy = g.center
            d = math.hypot(gx - cx, gy - cy)
            if d < best_d:
                best_d = d
                best_g = g

        if best_g is None or best_d > 35.0:
            continue

        ordered = member_pca_order(best_g)

        # Only use groups where each path encodes exactly one character.
        # Single-member groups are compound paths (whole number as one path) —
        # we cannot decompose them into individual digit templates.
        if len(ordered) != len(chars):
            continue

        for drawing, char in zip(ordered, chars):
            contour = path_to_contour(drawing)
            if contour is None or len(contour) < 5:
                continue
            hu = compute_hu(contour)
            templates.setdefault(char, []).append((hu, contour))

    return templates


def recognize_digit_by_shape(
    drawing: dict,
    templates: DigitTemplates,
) -> tuple[str, float] | None:
    """Match a single filled path against digit templates using Hu moments.

    Returns ``(char, confidence)`` where confidence is in [0, 1], or ``None``
    when no template scores well enough.  The match uses L2 distance in the
    log-Hu space; the 7th moment (h7) is slightly skew-sensitive, which helps
    distinguish ``6`` from ``9`` when they appear at the same rotation angle in
    the training data.
    """
    contour = path_to_contour(drawing)
    if contour is None or len(contour) < 5:
        return None

    unknown_hu = compute_hu(contour)

    best_char: str | None = None
    best_dist = float("inf")

    for char, entries in templates.items():
        for tmpl_hu, _ in entries:
            dist = float(np.linalg.norm(unknown_hu - tmpl_hu))
            if dist < best_dist:
                best_dist = dist
                best_char = char

    if best_char is None:
        return None

    # Reject if the best match is still very far — likely a path type we have
    # no template for (a chain arrow, a dot, etc.).
    _MAX_HU_DIST = 2.0
    if best_dist > _MAX_HU_DIST:
        return None

    confidence = max(0.0, 1.0 - best_dist / _MAX_HU_DIST)
    return (best_char, confidence)


def fix_6_9(
    char: str,
    char_bbox: fitz.Rect,
    cluster_angle_deg: float,
    cluster_bbox: fitz.Rect,
) -> str:
    """Disambiguate '6' vs '9' using the text-line orientation.

    Hu moments h1-h6 are rotation-invariant, so a ``6`` template learned at 0°
    can match a ``9`` at 180°.  This heuristic resolves the ambiguity using the
    cluster's PCA angle to define the baseline normal ("up" direction in PDF
    space).  The character whose closed loop faces "up" is '6'; the one whose
    closed loop faces "down" is '9'.

    In Tamil Nadu FMBs the closed loop of '6' is at the bottom of the glyph and
    the tail sweeps up, so the glyph's centre of mass is biased toward the
    closed-loop side.  We approximate this by checking whether the character
    center is displaced toward the "down" side of the cluster baseline (→ '9')
    or the "up" side (→ '6').  When both characters are present in the same
    cluster the relative displacement is more reliable than an absolute sign.
    """
    if char not in ("6", "9"):
        return char

    # Perpendicular to the text baseline = "up" direction in PDF y-down space.
    angle_rad = math.radians(cluster_angle_deg)
    # Rotate 90° CCW from baseline direction.
    up_x = -math.sin(angle_rad)
    up_y = math.cos(angle_rad)

    cx = (char_bbox.x0 + char_bbox.x1) / 2
    cy = (char_bbox.y0 + char_bbox.y1) / 2
    cluster_cx = (cluster_bbox.x0 + cluster_bbox.x1) / 2
    cluster_cy = (cluster_bbox.y0 + cluster_bbox.y1) / 2

    dot = (cx - cluster_cx) * up_x + (cy - cluster_cy) * up_y
    # Positive dot = character center is in the "up" half = closed loop at bottom = '6'.
    # Negative dot = character center is in the "down" half = closed loop at top = '9'.
    return "6" if dot >= 0 else "9"
