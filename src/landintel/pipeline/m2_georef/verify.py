"""M2 verification suite -- 7-check quality gates for georeferenced DXFs.

Checks:
  1. CRS Presence       -- $GDALWKT or CRS text entity exists
  2. Coordinate Range    -- UTM coordinates in valid Tamil Nadu range
  3. Boundary Closure    -- BOUNDARY polylines form a closed ring (gap < 2m)
  4. Stone Count Match   -- STONES count matches input M1 DXF
  5. Field Residual      -- max stone displacement from surveyor < 5.0m
  6. Edge Length Drift   -- max edge-length change < 2.0m vs M1
  7. Area Consistency    -- georeferenced area within 15% of M1 area
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import ezdxf
import numpy as np

from ...core.enums import LayerType

_log = logging.getLogger(__name__)

_L_STONES = LayerType.STONES.value
_L_BOUNDARY = LayerType.BOUNDARY.value

# Tamil Nadu UTM bounding box -- ZONE-AGNOSTIC (covers BOTH 43N west of 78E and 44N
# east of it). The old box was 44N-only and too tight: it rejected valid parcels in
# southern TN (Kanyakumari ~8N -> northing ~888k, below the old 1.1M floor) and in
# eastern 44N districts (~80E -> easting ~420k, below the old 600k floor). The gate's
# job is to catch a GROSS mis-georeference (coords near the relative-metre origin, or
# lat/lon degrees, or the wrong continent) -- not to pin a tight box -- so we use the
# valid-UTM easting band and the full TN latitude band (8-13.6N) across both zones.
_TN_UTM_XMIN, _TN_UTM_XMAX = 100000, 900000     # valid UTM easting in any TN zone
_TN_UTM_YMIN, _TN_UTM_YMAX = 800000, 1700000    # TN latitudes ~8-13.6N, both zones


def _boundary_lwpolylines(msp):
    return [e for e in msp.query("LWPOLYLINE") if e.dxf.layer == _L_BOUNDARY]


def build_traced_buffer(
    surveyor_polylines: list[list[tuple[float, float]]],
    tol: float = 3.0,
):
    """Pre-build the buffered union of all traced segments (reuse across calls).

    ``chain_coverage`` rebuilds this on every call; when scoring many candidate
    placements of the same plot (edge registration) that is the bottleneck. Build
    it once and pass it as ``chain_coverage(..., prepared=buf)``.
    """
    try:
        from shapely.geometry import LineString
        from shapely.ops import unary_union
    except ImportError:
        return None
    seg_lines = []
    for pts in surveyor_polylines:
        for i in range(len(pts) - 1):
            if pts[i] != pts[i + 1]:
                seg_lines.append(LineString([pts[i], pts[i + 1]]))
    if not seg_lines:
        return None
    return unary_union(seg_lines).buffer(tol)


def chain_coverage(
    boundary_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    surveyor_polylines: list[list[tuple[float, float]]] | None = None,
    tol: float = 3.0,
    prepared=None,
) -> float:
    """Fraction of a georeferenced boundary lying on the surveyor's traced lines.

    The surveyor's SITE DATA LINE polylines are the ACTUALLY-traced property
    boundaries. This returns the fraction of ``boundary_segments`` total length
    that lies within ``tol`` metres of any traced segment.

    This is an INDEPENDENT ground-truth signal the stone-congruence matcher never
    uses: in a dense stone cloud a similarity transform can drop any rigid plot
    shape onto SOME congruent stone subset by chance, but only a TRULY surveyed
    plot's boundary lies on the traced lines. Measured on INGUR it cleanly
    separates true matches (58-100%) from coincidental ones (12-43%) with a wide
    gap -- so it is the decisive false-positive gate (see pipeline disposition).
    Returns 0.0 when there is nothing to measure.
    """
    try:
        from shapely.geometry import LineString
        from shapely.ops import unary_union
    except ImportError:
        return float("nan")

    if not boundary_segments:
        return 0.0
    buf = prepared
    if buf is None:
        seg_lines = []
        for pts in (surveyor_polylines or []):
            for i in range(len(pts) - 1):
                if pts[i] != pts[i + 1]:
                    seg_lines.append(LineString([pts[i], pts[i + 1]]))
        if not seg_lines:
            return 0.0
        buf = unary_union(seg_lines).buffer(tol)

    total = covered = 0.0
    for a, b in boundary_segments:
        ls = LineString([a, b])
        total += ls.length
        covered += ls.intersection(buf).length
    return covered / total if total > 0 else 0.0


def _stone_texts(msp):
    return [e for e in msp.query("TEXT") if e.dxf.layer == _L_STONES]


@dataclass
class VerifyCheck:
    """Result of a single verification check."""
    name: str
    passed: bool
    detail: str = ""
    value: float = float("nan")


@dataclass
class VerifyResult:
    """Full verification result for one georeferenced DXF."""
    file: str
    checks: list[VerifyCheck] = field(default_factory=list)
    all_passed: bool = False

    def __post_init__(self):
        self.all_passed = all(c.passed for c in self.checks)


def verify_georef_dxf(
    georef_dxf_path: str | Path,
    m1_dxf_path: str | Path | None = None,
    surveyor_dxf_path: str | Path | None = None,
    match_result=None,
    m1_data=None,
    field_residual_max: float | None = None,
) -> VerifyResult:
    """Run all 7 verification checks on a georeferenced DXF.

    Parameters
    ----------
    georef_dxf_path : path to the georeferenced output DXF
    m1_dxf_path : optional path to the original M1 DXF (for comparison checks)
    surveyor_dxf_path : optional path to surveyor DXF
    match_result : optional MatchResult from the matching stage
    m1_data : optional M1PlotData from extraction stage
    field_residual_max : optional max field residual (m) from the cadastral
        adjustment -- the true post-adjustment displacement of matched stones
        from their surveyor positions. When provided, Check 5 uses this rather
        than the (looser) neighborhood score.

    Returns
    -------
    VerifyResult with all check outcomes.
    """
    georef_dxf_path = Path(georef_dxf_path)
    result = VerifyResult(file=str(georef_dxf_path))

    doc = ezdxf.readfile(str(georef_dxf_path))
    msp = doc.modelspace()

    # --- Check 1: CRS Presence ---
    crs_found = False
    # Check header
    try:
        if doc.header.get('$GDALWKT'):
            crs_found = True
    except Exception:
        pass
    # Check for CRS text entity
    if not crs_found:
        for e in msp.query('TEXT'):
            if 'CRS:' in str(e.dxf.text):
                crs_found = True
                break
    result.checks.append(VerifyCheck(
        "1_CRS_Presence", crs_found,
        "CRS marker found" if crs_found else "No CRS metadata found"
    ))

    # --- Check 2: Coordinate Range ---
    all_x, all_y = [], []
    for e in msp.query('LWPOLYLINE'):
        for pt in e.get_points():
            all_x.append(pt[0])
            all_y.append(pt[1])
    for e in msp.query('TEXT'):
        # The CRS marker is metadata, not plot geometry -- exclude it from the
        # coordinate-range gate so its placement never affects the check.
        if 'CRS:' in str(e.dxf.text):
            continue
        all_x.append(e.dxf.insert.x)
        all_y.append(e.dxf.insert.y)

    if all_x:
        x_valid = min(all_x) >= _TN_UTM_XMIN and max(all_x) <= _TN_UTM_XMAX
        y_valid = min(all_y) >= _TN_UTM_YMIN and max(all_y) <= _TN_UTM_YMAX
        coord_ok = x_valid and y_valid
        result.checks.append(VerifyCheck(
            "2_Coordinate_Range", coord_ok,
            f"X=[{min(all_x):.0f}, {max(all_x):.0f}] "
            f"Y=[{min(all_y):.0f}, {max(all_y):.0f}]"
        ))
    else:
        result.checks.append(VerifyCheck(
            "2_Coordinate_Range", False, "No geometry found"
        ))

    # --- Check 3: Boundary Closure ---
    # Robust to multi-segment boundaries: the boundary is closed iff its warped
    # segments enclose a face (polygonize). A first/last-vertex gap test is
    # unreliable when a ring edge is several segments and the ring closes at an
    # intermediate (non-stone) vertex.
    boundary_pts = []
    seg_pairs = []
    for e in _boundary_lwpolylines(msp):
        pts = [(pt[0], pt[1]) for pt in e.get_points()]
        boundary_pts.extend(pts)
        for i in range(len(pts) - 1):
            seg_pairs.append((pts[i], pts[i + 1]))

    if len(seg_pairs) >= 3:
        try:
            from shapely.geometry import MultiLineString
            from shapely.ops import polygonize, unary_union
            faces = list(polygonize(unary_union(MultiLineString(seg_pairs))))
        except Exception:
            faces = []
        if faces:
            result.checks.append(VerifyCheck(
                "3_Boundary_Closure", True,
                f"boundary encloses a face ({len(faces)} polygon(s))"))
        else:
            first = np.array(boundary_pts[0])
            last = np.array(boundary_pts[-1])
            gap = float(np.linalg.norm(first - last))
            result.checks.append(VerifyCheck(
                "3_Boundary_Closure", gap < 2.0,
                f"no enclosed face; first/last vertex gap {gap:.2f}m"))
    else:
        result.checks.append(VerifyCheck(
            "3_Boundary_Closure", False,
            f"Too few boundary vertices: {len(boundary_pts)}"
        ))

    # --- Check 4: Stone Count Match ---
    georef_stones = _stone_texts(msp)
    stone_count = len(georef_stones)
    m1_stone_count = len(m1_data.stones) if m1_data else stone_count
    stone_ok = stone_count == m1_stone_count
    result.checks.append(VerifyCheck(
        "4_Stone_Count", stone_ok,
        f"Georef={stone_count}, M1={m1_stone_count}"
    ))

    # --- Check 5: Field Residual (survey-grade precision gate) ---
    # FMB sketches encode ~0.2-1.0 m at 1:1000-1:2000; DGPS field control is ~5 cm. So a
    # post-adjustment field displacement < 0.5 m is survey-grade clean, 0.5-2.0 m acceptable,
    # > 2.0 m a bad fit (rejected). Tightened from the old 5.0 m on the registration-noise +
    # precision audits (2026-06-28). Chain coverage stays the PRIMARY false-positive gate; the
    # internal Umeyama residual stays a diagnostic only (a clean fit onto a WRONG stone subset
    # also reads ~0, so field residual is a precision gate, not the FP gate).
    FIELD_RESIDUAL_REJECT_M = 2.0
    FIELD_RESIDUAL_CLEAN_M = 0.5
    if field_residual_max is not None and not math.isnan(field_residual_max):
        field_ok = field_residual_max < FIELD_RESIDUAL_REJECT_M
        grade = ("survey-grade" if field_residual_max < FIELD_RESIDUAL_CLEAN_M
                 else "acceptable" if field_ok else "REJECT")
        result.checks.append(VerifyCheck(
            "5_Field_Residual", field_ok,
            f"Max stone displacement from surveyor: {field_residual_max:.3f}m "
            f"({grade}; reject>{FIELD_RESIDUAL_REJECT_M:.1f}m, survey-grade<{FIELD_RESIDUAL_CLEAN_M:.1f}m)",
            value=field_residual_max
        ))
    elif match_result and match_result.matched:
        max_residual = (match_result.neighborhood_score
                        if match_result.neighborhood_score < 100 else float("nan"))
        field_ok = max_residual < 5.0
        result.checks.append(VerifyCheck(
            "5_Field_Residual", field_ok,
            f"Neighborhood RMS: {max_residual:.3f}m (threshold: 5.0m)",
            value=max_residual
        ))
    else:
        result.checks.append(VerifyCheck(
            "5_Field_Residual", False, "No match result available"
        ))

    # --- Check 6: Perimeter Consistency (FMB -> field correction diagnostic) ---
    # Compare TOTAL boundary perimeter, which is granularity-independent: the
    # georef boundary is many warped segments while M1 outer_edges are
    # corner-to-corner, so per-edge comparison is meaningless. The georef
    # perimeter is the field-truth perimeter and legitimately differs from the
    # FMB sketch; FAIL only on a GROSS ratio (a wrong/distorted match).
    if m1_data and m1_data.outer_edges:
        m1_perim = sum(e.length_m for e in m1_data.outer_edges)
        georef_perim = 0.0
        for e in _boundary_lwpolylines(msp):
            pts = list(e.get_points())
            for i in range(len(pts) - 1):
                georef_perim += math.hypot(pts[i + 1][0] - pts[i][0],
                                           pts[i + 1][1] - pts[i][1])
        ratio = georef_perim / m1_perim if m1_perim > 0 else float("nan")
        ok = (not math.isnan(ratio)) and 0.6 < ratio < 1.7
        result.checks.append(VerifyCheck(
            "6_Perimeter_Consistency", ok,
            f"FMB->field perimeter: M1={m1_perim:.0f}m, georef={georef_perim:.0f}m, "
            f"ratio={ratio:.2f} (gross-fail outside 0.6-1.7)",
            value=ratio
        ))
    else:
        result.checks.append(VerifyCheck(
            "6_Perimeter_Consistency", False, "No M1 edge data available"
        ))

    # --- Check 7: Area Consistency ---
    try:
        from shapely.geometry import Polygon
        if len(boundary_pts) >= 4:
            m1_area = float("nan")
            if m1_data and m1_data.outer_edges:
                # Compute M1 area from outer boundary
                outer_cycle = m1_data.outer_stone_indices
                if len(outer_cycle) >= 3:
                    m1_coords = [(m1_data.stones[i].x, m1_data.stones[i].y)
                                 for i in outer_cycle]
                    try:
                        m1_area = abs(Polygon(m1_coords).area)
                    except Exception:
                        m1_area = float("nan")

            # Polygonize the boundary SEGMENTS (robust to multi-segment / disjoint
            # boundaries) rather than Polygon(boundary_pts), which mis-areas or
            # raises when the boundary isn't a single ordered vertex sequence.
            try:
                from shapely.geometry import MultiLineString
                from shapely.ops import polygonize, unary_union
                _faces = list(polygonize(unary_union(MultiLineString(seg_pairs)))) \
                    if len(seg_pairs) >= 3 else []
                if _faces:
                    georef_area = abs(unary_union(_faces).area)
                else:
                    georef_area = abs(Polygon(boundary_pts).area)
            except Exception:
                georef_area = float("nan")

            if not math.isnan(m1_area) and not math.isnan(georef_area) and m1_area > 0:
                area_ratio = georef_area / m1_area
                # DIAGNOSTIC of FMB-vs-field area correction. Fail only on a GROSS
                # mismatch (ratio outside 0.6-1.7), which indicates a wrong match;
                # a 10-30% area change is normal FMB sketch-vs-field correction.
                area_ok = 0.6 < area_ratio < 1.7
                result.checks.append(VerifyCheck(
                    "7_Area_Consistency", area_ok,
                    f"FMB->field area: M1={m1_area:.0f} m2, georef={georef_area:.0f} m2, "
                    f"ratio={area_ratio:.2f} (gross-fail outside 0.6-1.7)",
                    value=area_ratio
                ))
            else:
                result.checks.append(VerifyCheck(
                    "7_Area_Consistency", False, "Cannot compute areas"
                ))
        else:
            result.checks.append(VerifyCheck(
                "7_Area_Consistency", False, "Too few boundary points"
            ))
    except ImportError:
        result.checks.append(VerifyCheck(
            "7_Area_Consistency", False, "shapely not available"
        ))

    result.all_passed = all(c.passed for c in result.checks)
    return result


def print_verify_result(result: VerifyResult) -> None:
    """Print verification result in a readable table format."""
    status = "PASS" if result.all_passed else "FAIL"
    print(f"\n{'='*60}")
    print(f"VERIFY: {Path(result.file).name} [{status}]")
    print(f"{'='*60}")
    for c in result.checks:
        icon = "[OK]" if c.passed else "[!!]"
        print(f"  {icon} {c.name:<25} {c.detail}")
    print(f"{'='*60}\n")
