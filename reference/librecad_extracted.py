"""
librecad_extracted.py — Useful algorithms extracted from LibreCAD (GPLv2) for M2 pipeline.

Ported from C++ to Python. Source: https://github.com/LibreCAD/LibreCAD/

Original C++ files extracted and ported:
  - librecad/src/lib/math/rs_math.h, rs_math.cpp
  - librecad/src/lib/math/lc_linemath.h, lc_linemath.cpp
  - librecad/src/lib/engine/rs_vector.h, rs_vector.cpp
  - librecad/src/lib/engine/document/container/lc_looputils.h, lc_looputils.cpp
  - librecad/src/lib/engine/document/container/lc_pathbuilder.h, lc_pathbuilder.cpp
  - librecad/src/lib/filters/rs_filterdxf1.h

Categories of utilities extracted:
  1. FLOATING POINT COMPARISON (ULP-aware, tolerance-safe)
  2. ANGLE UTILITIES (normalize, between-test, difference, readable)
  3. 2D VECTOR OPERATIONS (rotate, mirror, shear, project, lerp)
  4. LINE GEOMETRY (intersection, projection, point-position, parallel, ray-test)
  5. CONVEX HULL (Graham scan)
  6. POLYGON AREA (shoelace with signed orientation)
  7. DXF GROUP CODES (entity type codes for M1 DXF parsing)
  8. LOOP TOPOLOGY (area with holes, point-in-polygon, winding number)
  9. POLYNOMIAL SOLVERS (quadratic, cubic, quartic — for arc/ellipse intersections)
 10. LINEAR SOLVER (Gauss-Jordan elimination)
 11. LOOP EXTRACTOR (extract closed loops from edge graph — for parcel polygon reconstruction)
 12. LOOP SORTER (hierarchical containment for outer/inner/holes)
 13. ELLIPSE PARAMETRICS (point and tangent on ellipse)
 14. POLYGON UTILITIES (orientation, Douglas-Peucker simplification, bounding box, self-intersection)
 15. MEANINGFUL VALUE HELPERS (tolerance-safe value checks from lc_linemath)
 16. BULGE TO ARC CONVERSION (DXF LWPOLYLINE bulge segments)
 17. SECOND MOMENTS OF AREA (Ixx, Iyy, Ixy via Green's theorem)

License: GPL-2.0 (inherited from LibreCAD). This module is for reference;
  the M2 pipeline already uses numpy/shapely for most ops, but these algorithms
  are useful for edge cases and DXF parsing.
"""

from __future__ import annotations

import math
import cmath
from typing import List, Optional, Tuple, Dict, Any

# ============================================================
# 1. CONSTANTS & TOLERANCES (from rs.h)
# ============================================================
RS_TOLERANCE = 1.0e-10       # General floating-point equality tolerance
RS_TOLERANCE2 = 1.0e-20      # Squared tolerance
RS_TOLERANCE_ANGLE = 1.0e-8  # Angle tolerance in radians (~5.7e-7 deg)
RS_MAXDOUBLE = 1.7976931348623157e+308

# DXF Entity Types (from rs.h RS2::EntityType) — used by M1 DXF reader
DXF_ENTITY_POINT = 1
DXF_ENTITY_LINE = 2
DXF_ENTITY_POLYLINE = 3
DXF_ENTITY_ARC = 4
DXF_ENTITY_CIRCLE = 5
DXF_ENTITY_ELLIPSE = 6
DXF_ENTITY_INSERT = 7
DXF_ENTITY_TEXT = 8
DXF_ENTITY_MTEXT = 9
DXF_ENTITY_SOLID = 10
DXF_ENTITY_LWPOLYLINE = 19

# ============================================================
# 2. FLOATING POINT UTILITIES (from rs_math)
# ============================================================

def ulp(x: float) -> float:
    """Unit in the Last Place — the gap between this float and the next representable float.
    From LibreCAD rs_math.h template. Useful for tolerance-aware comparisons."""
    if math.isfinite(x):
        if math.copysign(1.0, x) < 0:
            return x - math.nextafter(x, -math.inf)
        else:
            return math.nextafter(x, math.inf) - x
    return abs(x) * 1e-15


def feq(a: float, b: float, tol: float = 0.0) -> bool:
    """Floating-point equality test (from RS_Math::equal).
    If tol=0, uses 2*ULP of the larger value as tolerance."""
    if tol == 0.0:
        tol = 2.0 * max(ulp(a), ulp(b))
    return abs(a - b) <= tol


def fneq(a: float, b: float, tol: float = 0.0) -> bool:
    """Floating-point inequality test (from RS_Math::notEqual)."""
    return not feq(a, b, tol)


def fless(a: float, b: float) -> bool:
    """a <= b within floating-point tolerance (from RS_Math::less)."""
    return a <= b + 2.0 * ulp(b)


def fin_between(x: float, a: float, b: float) -> bool:
    """Test if x is between a and b (inclusive, tolerance-aware).
    From RS_Math::inBetween."""
    return fless(x, max(a, b)) and fless(min(a, b), x)


def is_meaningful(value: float) -> bool:
    """True if |value| >= RS_TOLERANCE. From LC_LineMath::isMeaningful."""
    return abs(value) >= RS_TOLERANCE


def is_not_meaningful(value: float) -> bool:
    """True if |value| < RS_TOLERANCE. From LC_LineMath::isNotMeaningful."""
    return abs(value) < RS_TOLERANCE


def is_meaningful_angle(value: float) -> bool:
    """True if |value| >= RS_TOLERANCE_ANGLE. From LC_LineMath::isMeaningfulAngle."""
    return abs(value) >= RS_TOLERANCE_ANGLE


def is_same_angle(a1: float, a2: float) -> bool:
    """True if two angles are the same within angle tolerance.
    From LC_LineMath::isSameAngle."""
    return abs(a1 - a2) < RS_TOLERANCE_ANGLE


def is_meaningful_distance(x1, y1, x2, y2) -> bool:
    """True if distance between two points >= RS_TOLERANCE.
    From LC_LineMath::isMeaningfulDistance."""
    return math.hypot(x2 - x1, y2 - y1) >= RS_TOLERANCE


def get_meaningful(candidate: float, replacement: float) -> float:
    """Return replacement if |candidate| < RS_TOLERANCE. From LC_LineMath::getMeaningful."""
    return replacement if abs(candidate) < RS_TOLERANCE else candidate


def get_meaningful_positive(candidate: float, replacement: float) -> float:
    """Return replacement if candidate < RS_TOLERANCE. From LC_LineMath::getMeaningfulPositive."""
    return replacement if candidate < RS_TOLERANCE else candidate


# ============================================================
# 3. ANGLE UTILITIES (from rs_math)
# ============================================================

def correct_angle(a: float) -> float:
    """Correct angle to [0, 2*PI). From RS_Math::correctAngle.
    Uses remainder-based approach matching the C++ implementation."""
    TWO_PI = 2.0 * math.pi
    return math.fmod(math.pi + math.remainder(a - math.pi, TWO_PI), TWO_PI)


def correct_angle_pm_pi(a: float) -> float:
    """Correct angle to [-PI, +PI). From RS_Math::correctAnglePlusMinusPi."""
    return math.remainder(a, 2.0 * math.pi)


def correct_angle_0_to_pi(a: float) -> float:
    """Correct angle to unsigned [0, PI). From RS_Math::correctAngle0ToPi."""
    return abs(math.remainder(a, 2.0 * math.pi))


def rad2deg(a: float) -> float:
    """Radians to degrees. From RS_Math::rad2deg."""
    return 180.0 / math.pi * a


def deg2rad(a: float) -> float:
    """Degrees to radians. From RS_Math::deg2rad."""
    return math.pi / 180.0 * a


def rad2gra(a: float) -> float:
    """Radians to gradians. From RS_Math::rad2gra."""
    return 200.0 / math.pi * a


def gra2rad(a: float) -> float:
    """Gradians to radians. From RS_Math::gra2rad."""
    return math.pi / 200.0 * a


def is_angle_between(a: float, amin: float, amax: float, reversed: bool = False) -> bool:
    """Test if angle a is between amin and amax (all radians).
    From RS_Math::isAngleBetween. Uses correctAngle-based range test."""
    if reversed:
        amin, amax = amax, amin
    # If the arc spans the full circle (or nearly so), any angle is between
    if get_angle_difference_unsigned(amax, amin) < RS_TOLERANCE_ANGLE:
        return True
    tol = 0.5 * RS_TOLERANCE_ANGLE
    diff0 = correct_angle(amax - amin) + tol
    return diff0 >= correct_angle(a - amin) or diff0 >= correct_angle(amax - a)


def get_angle_difference(a1: float, a2: float, reversed: bool = False) -> float:
    """Angular difference: the angle to add to a1 to reach a2.
    Always positive and less than 2*PI. From RS_Math::getAngleDifference."""
    if reversed:
        a1, a2 = a2, a1
    return correct_angle(a2 - a1)


def get_angle_difference_unsigned(a1: float, a2: float) -> float:
    """Minimum unsigned angular difference. From RS_Math::getAngleDifferenceU.
    Returns value in [0, PI]."""
    return correct_angle_0_to_pi(a1 - a2)


def is_angle_readable(angle: float) -> bool:
    """True if angle produces readable text (quadrant 1 & 4).
    From RS_Math::isAngleReadable."""
    tolerance = 0.001
    if angle > math.pi / 2:
        return abs(math.remainder(angle, 2.0 * math.pi)) < (math.pi / 2 - tolerance)
    else:
        return abs(math.remainder(angle, 2.0 * math.pi)) < (math.pi / 2 + tolerance)


def make_angle_readable(angle: float, readable: bool = True) -> float:
    """Adjust angle so text at that angle is readable from bottom/right.
    From RS_Math::makeAngleReadable."""
    ret = correct_angle(angle)
    cor = is_angle_readable(ret) ^ readable
    if cor:
        ret = correct_angle(angle + math.pi)
    return ret


def is_same_direction(dir1: float, dir2: float, tol: float = RS_TOLERANCE_ANGLE) -> bool:
    """True if two directions point the same way. From RS_Math::isSameDirection."""
    return get_angle_difference_unsigned(dir1, dir2) < tol


def calculate_angles(angle: float) -> Tuple[float, float, float, float]:
    """Compute complementary, supplementary, and alternate angles.
    From RS_Math::calculateAngles.
    Returns (angle, complementary, supplementary, alt)."""
    angle = correct_angle_0_to_pi(angle)
    complementary = math.pi / 2 - angle
    supplementary = math.pi - angle
    alt = math.pi + supplementary
    return (angle, complementary, supplementary, alt)


# ============================================================
# 4. 2D VECTOR UTILITIES (from rs_vector)
# ============================================================

def polar(rho: float, theta: float) -> Tuple[float, float]:
    """Polar to Cartesian. From RS_Vector::polar."""
    return (rho * math.cos(theta), rho * math.sin(theta))


def vec_angle(vx: float, vy: float) -> float:
    """Angle of vector from origin. From RS_Vector::angle."""
    return correct_angle(math.atan2(vy, vx))


def vec_angle_to(x1, y1, x2, y2):
    """Angle from (x1,y1) to (x2,y2). From RS_Vector::angleTo."""
    return vec_angle(x2 - x1, y2 - y1)


def vec_distance(x1, y1, x2, y2) -> float:
    """Euclidean distance. From RS_Vector::distanceTo."""
    return math.hypot(x2 - x1, y2 - y1)


def vec_magnitude(vx: float, vy: float) -> float:
    return math.hypot(vx, vy)


def vec_normalize(vx: float, vy: float) -> Tuple[float, float]:
    """Normalize to unit vector. From RS_Vector::normalize."""
    m = math.hypot(vx, vy)
    if m > RS_TOLERANCE:
        return (vx / m, vy / m)
    return (vx, vy)


def vec_dot(x1, y1, x2, y2) -> float:
    """2D dot product. From RS_Vector::dotP."""
    return x1 * x2 + y1 * y2


def vec_cross_2d(x1, y1, x2, y2) -> float:
    """2D cross product (z-component). From RS_Vector::crossP for 2D."""
    return x1 * y2 - y1 * x2


def vec_rotate(x, y, angle_rad):
    """Rotate (x,y) around origin by angle_rad. From RS_Vector::rotate(RS_Vector).
    Uses the efficient cos/sin decomposition."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (x * c - y * s, x * s + y * c)


def vec_rotate_around(x, y, cx, cy, angle_rad):
    """Rotate (x,y) around (cx,cy). From RS_Vector::rotate(center, angle)."""
    dx, dy = x - cx, y - cy
    rx, ry = vec_rotate(dx, dy, angle_rad)
    return (cx + rx, cy + ry)


def vec_mirror(x, y, ax1, ay1, ax2, ay2):
    """Mirror (x,y) across the axis defined by (ax1,ay1)-(ax2,ay2).
    From RS_Vector::mirror(axisPoint1, axisPoint2).
    Formula: proj = A + dir * dot(P-A, dir) / |dir|^2; result = 2*proj - P."""
    dx, dy = ax2 - ax1, ay2 - ay1
    a_sq = dx * dx + dy * dy
    if a_sq < RS_TOLERANCE2:
        return (x, y)  # degenerate axis
    apx, apy = x - ax1, y - ay1
    t = (apx * dx + apy * dy) / a_sq
    px, py = ax1 + t * dx, ay1 + t * dy  # projection point
    return (2 * px - x, 2 * py - y)


def vec_lerp(x1, y1, x2, y2, t: float):
    """Linear interpolation. From RS_Vector::lerp."""
    return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)


def vec_shear(x, y, k: float):
    """Shear transform: x = x + k*y. From RS_Vector::shear."""
    return (x + k * y, y)


def relative_point(sx, sy, distance: float, angle_rad: float):
    """Point at distance and angle from start. From LC_LineMath::relativePoint."""
    ox, oy = polar(distance, angle_rad)
    return (sx + ox, sy + oy)


def end_of_line_segment(sx, sy, angle_deg: float, distance: float):
    """End point of line segment from start, angle (degrees), and distance.
    From LC_LineMath::getEndOfLineSegment."""
    return relative_point(sx, sy, distance, deg2rad(angle_deg))


def pos_in_line(sx, sy, ex, ey, px, py) -> float:
    """Position of point (px,py) on line segment (sx,sy)->(ex,ey) as a fraction.
    From RS_Vector::posInLine.
    Returns 0.0 at start, 1.0 at end, <0 behind, >1 beyond."""
    dex, dey = ex - sx, ey - sy
    dpx, dpy = px - sx, py - sy
    len_sq = dex * dex + dey * dey
    if len_sq < RS_TOLERANCE2:
        return vec_distance(sx, sy, px, py)
    return (dpx * dex + dpy * dey) / len_sq


# ============================================================
# 5. LINE GEOMETRY (from lc_linemath)
# ============================================================

def nearest_point_on_infinite_line(px, py, lx1, ly1, lx2, ly2):
    """Orthogonal projection of point onto infinite line. From LC_LineMath::getNearestPointOnInfiniteLine."""
    aex, aey = lx2 - lx1, ly2 - ly1
    apx, apy = px - lx1, py - ly1
    mag = math.hypot(aex, aey)
    if mag < RS_TOLERANCE:
        return None
    t = (apx * aex + apy * aey) / (mag * mag)
    return (lx1 + t * aex, ly1 + t * aey)


def line_line_intersection(s1, e1, s2, e2):
    """Intersection of two line segments (s1->e1) and (s2->e2).
    From LC_LineMath::getIntersectionLineLineFast.
    Returns (x, y) or None if parallel."""
    num = ((e2[0] - s2[0]) * (s1[1] - s2[1]) - (e2[1] - s2[1]) * (s1[0] - s2[0]))
    div = ((e2[1] - s2[1]) * (e1[0] - s1[0]) - (e2[0] - s2[0]) * (e1[1] - s1[1]))
    if abs(div) < RS_TOLERANCE:
        return None
    # Check that lines are not nearly parallel (angle check from C++)
    a1 = vec_angle_to(s1[0], s1[1], e1[0], e1[1])
    a2 = vec_angle_to(s2[0], s2[1], e2[0], e2[1])
    if abs(math.remainder(a1 - a2, math.pi)) < RS_TOLERANCE * 10:
        return None
    u = num / div
    return (s1[0] + u * (e1[0] - s1[0]), s1[1] + u * (e1[1] - s1[1]))


def infinite_line_line_intersection(s1, e1, s2, e2, offset_x=0.0, offset_y=0.0):
    """Intersection of infinite line (s1->e1) with finite line (s2->e2) bounded by offset.
    From LC_LineMath::getIntersectionInfiniteLineLineFast."""
    num = ((e2[0] - s2[0]) * (s1[1] - s2[1]) - (e2[1] - s2[1]) * (s1[0] - s2[0]))
    div = ((e2[1] - s2[1]) * (e1[0] - s1[0]) - (e2[0] - s2[0]) * (e1[1] - s1[1]))
    if abs(div) < RS_TOLERANCE:
        return None
    u = num / div
    xs = s1[0] + u * (e1[0] - s1[0])
    ys = s1[1] + u * (e1[1] - s1[1])
    xlo, xhi = min(s2[0], e2[0]) - offset_x, max(s2[0], e2[0]) + offset_x
    ylo, yhi = min(s2[1], e2[1]) - offset_y, max(s2[1], e2[1]) + offset_y
    if xlo <= xs <= xhi and ylo <= ys <= yhi:
        return (xs, ys)
    return None


def point_position(sx, sy, ex, ey, px, py) -> str:
    """Classify point position relative to directed line segment (sx,sy)->(ex,ey).
    From LC_LineMath::getPointPosition.
    Returns: 'LEFT', 'RIGHT', 'BEYOND', 'BEHIND', 'BETWEEN', 'ORIGIN', 'DESTINATION'."""
    ax, ay = ex - sx, ey - sy
    bx, by = px - sx, py - sy
    cross = ax * by - bx * ay
    if cross > 0:
        return "LEFT"
    if cross < 0:
        return "RIGHT"
    if (ax * bx < 0) or (ay * by < 0):
        return "BEHIND"
    if math.hypot(ax, ay) < math.hypot(bx, by):
        return "BEYOND"
    if feq(sx, px) and feq(sy, py):
        return "ORIGIN"
    if feq(ex, px) and feq(ey, py):
        return "DESTINATION"
    return "BETWEEN"


def has_line_segment_intersection(p0x, p0y, dx, dy, p2x, p2y, p3x, p3y):
    """Test if infinite line (p0, direction) crosses segment (p2, p3).
    From LC_LineMath::hasLineIntersection.
    Uses the normal-perpendicular method."""
    nx, ny = dy, -dx  # normal to direction
    rpx, rpy = p0x - p2x, p0y - p2y
    rx, ry = p3x - p2x, p3y - p2y
    denom = rx * nx + ry * ny
    if abs(denom) < RS_TOLERANCE:
        return False
    t = (rpx * nx + rpy * ny) / denom
    return 0.0 <= t <= 1.0


def has_line_rect_intersection(line_start, line_end, rect_min, rect_max):
    """Test if a line intersects an axis-aligned rectangle.
    From LC_LineMath::hasIntersectionLineRect.
    Tests intersection with both diagonals of the rectangle."""
    direction = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    if has_line_segment_intersection(
        line_start[0], line_start[1], direction[0], direction[1],
        rect_min[0], rect_min[1], rect_max[0], rect_max[1]
    ):
        return True
    return has_line_segment_intersection(
        line_start[0], line_start[1], direction[0], direction[1],
        rect_min[0], rect_max[1], rect_max[0], rect_min[1]
    )


def create_parallel(sx, sy, ex, ey, distance):
    """Create a line parallel to (sx,sy)->(ex,ey) at given distance.
    From LC_LineMath::createParallel. Returns (new_sx, new_sy, new_ex, new_ey)."""
    angle = vec_angle_to(sx, sy, ex, ey) + math.pi / 2
    dx, dy = distance * math.cos(angle), distance * math.sin(angle)
    return (sx + dx, sy + dy, ex + dx, ey + dy)


def are_lines_on_same_ray(l1s, l1e, l2s, l2e):
    """Check if two segments lie on the same ray. From LC_LineMath::areLinesOnSameRay."""
    a1 = correct_angle_0_to_pi(vec_angle_to(l1s[0], l1s[1], l1e[0], l1e[1]))
    a2 = correct_angle_0_to_pi(vec_angle_to(l1s[0], l1s[1], l2e[0], l2e[1]))
    a3 = correct_angle_0_to_pi(vec_angle_to(l1s[0], l1s[1], l2s[0], l2s[1]))
    tol = RS_TOLERANCE_ANGLE * 10
    return abs(a1 - a2) < tol and abs(a1 - a3) < tol


def find_point_on_circle(radius: float, arc_angle: float, cx: float, cy: float):
    """Point on circle at given radius and angle from center.
    From LC_LineMath::findPointOnCircle."""
    rx, ry = polar(radius, arc_angle)
    return (cx + rx, cy + ry)


def angle_for_3_points(p1, p_int, p2):
    """Angle at intersection between two edges. From LC_LineMath::angleFor3Points."""
    a1 = vec_angle_to(p_int[0], p_int[1], p1[0], p1[1])
    a2 = vec_angle_to(p_int[0], p_int[1], p2[0], p2[1])
    return correct_angle_0_to_pi(get_angle_difference(a1, a2))


def parallel_line_distance(sx1, sy1, sx2, sy2, px, py):
    """Distance from point (px,py) to infinite line through (sx1,sy1)-(sx2,sy2).
    Returns (distance, projection_point) or (inf, None)."""
    proj = nearest_point_on_infinite_line(px, py, sx1, sy1, sx2, sy2)
    if proj is None:
        return (float("inf"), None)
    d = vec_distance(px, py, proj[0], proj[1])
    return (d, proj)


# ============================================================
# 6. CONVEX HULL (from lc_linemath, Graham's scan)
# ============================================================

def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Graham's scan convex hull. From LC_LineMath::convexHull.
    Input: list of (x, y) points. Output: ordered hull vertices (CCW)."""
    if len(points) <= 2:
        return list(points)
    # Find leftmost-lowest point (compareCoordinates from C++)
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    # Remove duplicates within tolerance
    hull = [pts[0]]
    for p in pts[1:]:
        if vec_distance(hull[-1][0], hull[-1][1], p[0], p[1]) > RS_TOLERANCE:
            hull.append(p)
    if len(hull) <= 2:
        return hull
    # Sort by angle from first point
    anchor = hull[0]
    rest = sorted(hull[1:], key=lambda p: math.atan2(p[1] - anchor[1], p[0] - anchor[0]))
    # Keep farthest for same angle
    unique = [anchor, rest[0]]
    for p in rest[1:]:
        prev_angle = math.atan2(unique[-1][1] - anchor[1], unique[-1][0] - anchor[0])
        curr_angle = math.atan2(p[1] - anchor[1], p[0] - anchor[0])
        if abs(curr_angle - prev_angle) < RS_TOLERANCE_ANGLE:
            if vec_distance(anchor[0], anchor[1], p[0], p[1]) > vec_distance(anchor[0], anchor[1], unique[-1][0], unique[-1][1]):
                unique[-1] = p
        else:
            unique.append(p)
    # Graham scan — keep only left turns (CCW test)
    stack = [unique[0], unique[1]]
    for p in unique[2:]:
        while len(stack) > 1:
            o = stack[-2]
            a = stack[-1]
            cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
            if cross >= -RS_TOLERANCE:
                break
            stack.pop()
        stack.append(p)
    return stack


# ============================================================
# 7. POLYGON AREA (Shoelace formula, signed)
# ============================================================

def polygon_area_signed(vertices: List[Tuple[float, float]]) -> float:
    """Signed area of a polygon using the shoelace formula.
    Positive = CCW, negative = CW. From LC_LoopUtils::getTotalArea logic."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return area / 2.0


def polygon_area(vertices: List[Tuple[float, float]]) -> float:
    """Unsigned polygon area."""
    return abs(polygon_area_signed(vertices))


def polygon_centroid(vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Centroid of a simple polygon."""
    n = len(vertices)
    if n == 0:
        return (0.0, 0.0)
    cx = cy = 0.0
    a = polygon_area_signed(vertices)
    if abs(a) < RS_TOLERANCE:
        return (sum(v[0] for v in vertices) / n, sum(v[1] for v in vertices) / n)
    for i in range(n):
        j = (i + 1) % n
        cross = vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1]
        cx += (vertices[i][0] + vertices[j][0]) * cross
        cy += (vertices[i][1] + vertices[j][1]) * cross
    factor = 1.0 / (6.0 * a)
    return (cx * factor, cy * factor)


# ============================================================
# 8. POINT-IN-POLYGON (winding number / ray casting)
# ============================================================

def point_in_polygon_winding(px, py, vertices: List[Tuple[float, float]]) -> bool:
    """Point-in-polygon using winding number. From LC_LoopUtils::getContainingDepth.
    Returns True if point is inside (odd winding number)."""
    n = len(vertices)
    if n < 3:
        return False
    winding = 0
    for i in range(n):
        j = (i + 1) % n
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if yi <= py:
            if yj > py:
                if (xj - xi) * (py - yi) - (px - xi) * (yj - yi) > 0:
                    winding += 1
        else:
            if yj <= py:
                if (xj - xi) * (py - yi) - (px - xi) * (yj - yi) < 0:
                    winding -= 1
    return winding % 2 != 0


def point_in_polygon_ray(px, py, vertices: List[Tuple[float, float]]) -> bool:
    """Point-in-polygon using ray casting. Simpler alternative to winding number."""
    n = len(vertices)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ============================================================
# 9. POLYNOMIAL SOLVERS (from rs_math)
# ============================================================

def quadratic_solver(a: float, b: float, c: float) -> List[float]:
    """Solve ax^2 + bx + c = 0. From RS_Math::quadraticSolver.
    Uses numerically stable formulation avoiding catastrophic cancellation.
    Returns list of real roots."""
    if abs(a) < RS_TOLERANCE:
        if abs(b) >= RS_TOLERANCE:
            return [-c / b]
        return []
    # Normalize: x^2 + (b/a)x + (c/a) = 0
    # Use long double precision approach from C++
    b_over_a = b / a
    c_over_a = c / a
    half_b = -0.5 * b_over_a
    b2 = half_b * half_b
    discriminant = b2 - c_over_a
    fc = abs(c_over_a)
    TOL = 1e-24

    if discriminant < 0.0:
        return []

    # Numerically stable radical computation
    if b2 >= fc:
        r = abs(half_b) * math.sqrt(1.0 - c_over_a / b2) if b2 > 0 else 0.0
    else:
        r = math.sqrt(fc) * math.sqrt(1.0 + b2 / fc)

    if r >= TOL * abs(half_b):
        if half_b >= 0.0:
            root1 = half_b + r
        else:
            root1 = half_b - r
        root2 = c_over_a / root1  # Vieta's formula
        return [root1, root2]
    else:
        return [half_b]


def cubic_solver(ce: List[float]) -> List[float]:
    """Solve x^3 + ce[0] x^2 + ce[1] x + ce[2] = 0. From RS_Math::cubicSolver.
    Uses Cardano's method with Newton-Raphson refinement.
    Returns list of real roots."""
    if len(ce) != 3:
        return []

    # Depressed cubic: t^3 + p*t + q = 0, where x = t - ce[0]/3
    shift = (1.0 / 3.0) * ce[0]
    p = ce[1] - shift * ce[0]
    q = ce[0] * ((2.0 / 27.0) * ce[0] * ce[0] - (1.0 / 3.0) * ce[1]) + ce[2]

    if abs(p) < 1.0e-75:
        root = math.cbrt(q) - shift
        return [root]

    discriminant = (1.0 / 27.0) * p * p * p + (1.0 / 4.0) * q * q

    ans = []
    if not math.isnan(discriminant) and discriminant >= 0:
        # One real root
        disc_sqrt = math.sqrt(max(0.0, discriminant))
        # Solve z^2 + q*z - p^3/27 = 0
        r = quadratic_solver(q, -p * p * p / 27.0)
        if not r:
            return []
        if len(r) < 2:
            u = math.cbrt(r[0])
        else:
            u = math.cbrt(r[0]) if q <= 0 else math.cbrt(r[1])
        v = (-1.0 / 3.0) * p / u
        ans.append(u + v - shift)
    else:
        # Three real roots via complex arithmetic
        u = complex(-0.5 * q, 0)
        u = ((-0.5 * u - cmath.sqrt(0.25 * u * u + p * p * p / 27.0)) ** (1.0 / 3.0))
        w = complex(-0.5, math.sqrt(3.0) / 2.0)
        ans.append((u - p / (3.0 * u) - shift).real)
        ans.append((u * w - p / (3.0 * u * w) - shift).real)
        ans.append((u / w - p * w / (3.0 * u) - shift).real)

    # Newton-Raphson refinement (from C++ code)
    for i, x0 in enumerate(ans):
        for _ in range(20):
            f = ((x0 + ce[0]) * x0 + ce[1]) * x0 + ce[2]
            df = (3.0 * x0 + 2.0 * ce[0]) * x0 + ce[1]
            if abs(df) > abs(f) + RS_TOLERANCE:
                x0 -= f / df
            else:
                break
        ans[i] = x0

    return ans


def quartic_solver(ce: List[float]) -> List[float]:
    """Solve x^4 + ce[0] x^3 + ce[1] x^2 + ce[2] x + ce[3] = 0.
    From RS_Math::quarticSolver.
    Uses Ferrari's method via resolvent cubic.
    Returns list of real roots."""
    if len(ce) != 4:
        return []

    # Depressed quartic: t^4 + p*t^2 + q*t + r = 0, where x = t - a/4
    shift = 0.25 * ce[0]
    shift2 = shift * shift
    a2 = ce[0] * ce[0]
    p = ce[1] - (3.0 / 8.0) * a2
    q = ce[2] + ce[0] * ((1.0 / 8.0) * a2 - 0.5 * ce[1])
    r = ce[3] - shift * ce[2] + (ce[1] - 3.0 * shift2) * shift2

    # Biquadratic case
    if q * q <= 1.0e-4 * RS_TOLERANCE * abs(p * r):
        discriminant = 0.25 * p * p - r
        if discriminant < -1.0e3 * RS_TOLERANCE:
            return []
        t2_0 = -0.5 * p - math.sqrt(abs(discriminant))
        t2_1 = -p - t2_0
        ans = []
        if t2_1 >= 0:
            ans.extend([math.sqrt(t2_1) - shift, -math.sqrt(t2_1) - shift])
        if t2_0 >= 0:
            ans.extend([math.sqrt(t2_0) - shift, -math.sqrt(t2_0) - shift])
        return ans

    # If r ≈ 0, factor out x=0 and solve cubic
    if abs(r) < 1.0e-75:
        cubic_ce = [p, q, 0.0]
        r3 = cubic_solver(cubic_ce)
        return [0.0 - shift] + [x - shift for x in r3]

    # Factor as (t^2 + u*t + v)(t^2 - u*t + w)
    # Resolvent cubic: y^3 + 2p*y^2 + (p^2 - 4r)*y - q^2 = 0
    cubic_ce = [2.0 * p, p * p - 4.0 * r, -q * q]
    r3 = cubic_solver(cubic_ce)
    if not r3:
        return []

    ans = []
    if len(r3) == 1:
        if r3[0] < 0:
            return []
        sqrtz0 = math.sqrt(r3[0])
        ce2 = [-sqrtz0, 0.5 * (p + r3[0]) + 0.5 * q / sqrtz0]
        r1 = quadratic_solver(1.0, ce2[0], ce2[1])
        if not r1:
            ce2 = [sqrtz0, 0.5 * (p + r3[0]) - 0.5 * q / sqrtz0]
            r1 = quadratic_solver(1.0, ce2[0], ce2[1])
        ans = [x - shift for x in r1]
    elif len(r3) >= 2 and r3[0] > 0 and r3[1] > 0:
        sqrtz0 = math.sqrt(r3[0])
        ce2_a = [-sqrtz0, 0.5 * (p + r3[0]) + 0.5 * q / sqrtz0]
        ce2_b = [sqrtz0, 0.5 * (p + r3[0]) - 0.5 * q / sqrtz0]
        r1 = quadratic_solver(1.0, ce2_a[0], ce2_a[1])
        r2 = quadratic_solver(1.0, ce2_b[0], ce2_b[1])
        ans = [x - shift for x in r1] + [x - shift for x in r2]

    # Newton-Raphson refinement
    for i, x0 in enumerate(ans):
        for _ in range(20):
            f = ((((x0 + ce[0]) * x0 + ce[1]) * x0 + ce[2]) * x0 + ce[3])
            df = (((4.0 * x0 + 3.0 * ce[0]) * x0 + 2.0 * ce[1]) * x0 + ce[2])
            if abs(df) > RS_TOLERANCE2:
                x0 -= f / df
            else:
                break
        ans[i] = x0

    return ans


def quartic_solver_full(ce: List[float]) -> List[float]:
    """Solve ce[4] x^4 + ce[3] x^3 + ce[2] x^2 + ce[1] x + ce[0] = 0.
    From RS_Math::quarticSolverFull. Full coefficient form."""
    if len(ce) != 5:
        return []
    if abs(ce[4]) < 1.0e-14:
        if abs(ce[3]) < 1.0e-14:
            if abs(ce[2]) < 1.0e-14:
                if abs(ce[1]) > 1.0e-14:
                    return [-ce[0] / ce[1]]
                return []
            return quadratic_solver(ce[2], ce[1], ce[0])
        return cubic_solver([ce[2] / ce[3], ce[1] / ce[3], ce[0] / ce[3]])
    ce2 = [ce[3] / ce[4], ce[2] / ce[4], ce[1] / ce[4], ce[0] / ce[4]]
    if abs(ce2[3]) <= RS_TOLERANCE * 1e-5:
        roots = cubic_solver(ce2[:3])
        roots.append(0.0)
        return roots
    return quartic_solver(ce2)


# ============================================================
# 10. LINEAR SOLVER (Gauss-Jordan elimination, from rs_math)
# ============================================================

def linear_solver(matrix: List[List[float]]) -> Optional[List[float]]:
    """Solve linear equation set Ax=b using Gauss-Jordan elimination.
    From RS_Math::linearSolver.
    @param matrix: augmented matrix [n x (n+1)], last column is RHS.
    @return: solution vector, or None if singular."""
    m_size = len(matrix)
    a_size = m_size + 1
    # Verify matrix size
    for row in matrix:
        if len(row) != a_size:
            return None

    # Deep copy
    mt = [row[:] for row in matrix]

    for i in range(m_size):
        # Find pivot
        imax = i
        cmax = abs(mt[i][i])
        for j in range(i + 1, m_size):
            if abs(mt[j][i]) > cmax:
                imax = j
                cmax = abs(mt[j][i])

        if cmax < RS_TOLERANCE:
            return None  # Singular matrix

        if imax != i:
            mt[i], mt[imax] = mt[imax], mt[i]

        # Normalize row i
        for k in range(i + 1, a_size):
            mt[i][k] /= mt[i][i]
        mt[i][i] = 1.0

        # Eliminate column i from all other rows
        for j in range(m_size):
            if j != i:
                a = mt[j][i]
                for k in range(i + 1, a_size):
                    mt[j][k] -= mt[i][k] * a
                mt[j][i] = 0.0

    return [mt[i][m_size] for i in range(m_size)]


# ============================================================
# 11. LOOP EXTRACTOR (from lc_looputils)
# ============================================================

ENDPOINT_TOLERANCE = 1e-8  # From LoopExtractor::ENDPOINT_TOLERANCE
VECTOR_KEY_SCALE = 1e8     # From makeVectorKey SCALE


def _make_vector_key(vx: float, vy: float) -> Tuple[int, int]:
    """Tolerance-aware integer key for endpoints, rounded to 1e-8 precision.
    From LoopExtractor's VectorKey / makeVectorKey."""
    return (round(vx * VECTOR_KEY_SCALE), round(vy * VECTOR_KEY_SCALE))


def extract_closed_loops(edges: List[dict]) -> List[List[Tuple[float, float]]]:
    """Extract closed loops from a set of edges.
    Ported from LC_LoopUtils::LoopExtractor::extract().

    Each edge is a dict with keys: 'start': (x,y), 'end': (x,y).
    Optionally 'id' for tracking.

    Algorithm:
    0. Mark all edges as unprocessed.
    1. Find the first edge on an outermost loop (extremal projection).
    2. Chain connected edges, choosing the outermost turn at junctions.
    3. Close the loop when end meets start within tolerance.
    4. Repeat until all edges processed.

    Assumptions:
    - Contours are closed loops (each edge's endpoints connect to other edges)
    - No self-intersection among contours
    - Each loop has the same number of edges as vertices (Euler characteristic 0)

    Returns: list of loops, each loop is a list of (x, y) vertices (CCW, positive area).
    """
    if not edges:
        return []

    # Build endpoint adjacency map
    endpoint_to_edges: Dict[Tuple[int, int], List[int]] = {}
    processed = [False] * len(edges)

    for i, e in enumerate(edges):
        s = e['start']
        end = e['end']
        # Skip degenerate zero-length edges
        if vec_distance(s[0], s[1], end[0], end[1]) <= ENDPOINT_TOLERANCE:
            processed[i] = True  # Mark degenerate as processed
            continue
        k1 = _make_vector_key(s[0], s[1])
        k2 = _make_vector_key(end[0], end[1])
        endpoint_to_edges.setdefault(k1, []).append(i)
        endpoint_to_edges.setdefault(k2, []).append(i)

    def _is_ccw(tri_a, tri_b, tri_c):
        """CCW test for 3 points. From isCounterClockwise in lc_linemath.cpp."""
        ax, ay = tri_b[0] - tri_a[0], tri_b[1] - tri_a[1]
        bx, by = tri_c[0] - tri_a[0], tri_c[1] - tri_a[1]
        cross = ax * by - bx * ay
        return cross >= RS_TOLERANCE

    results = []
    unprocessed = [i for i, p in enumerate(processed) if not p]

    while unprocessed:
        # findFirst: find extremal edge (leftmost-lowest, then sort by angle)
        # Simplified: just pick the first unprocessed edge
        first_idx = unprocessed[0]
        first_edge = edges[first_idx]

        # Build loop by chaining edges
        loop_indices = [first_idx]
        processed[first_idx] = True
        current_end = first_edge['end']
        target = first_edge['start']

        max_iter = len(unprocessed) + 2
        iteration = 0

        while vec_distance(current_end[0], current_end[1], target[0], target[1]) > ENDPOINT_TOLERANCE:
            iteration += 1
            if iteration > max_iter:
                break

            # Find connected unprocessed edges at current_end
            end_key = _make_vector_key(current_end[0], current_end[1])
            candidates = []
            for idx in endpoint_to_edges.get(end_key, []):
                if not processed[idx]:
                    candidates.append(idx)

            if not candidates:
                break

            if len(candidates) == 1:
                next_idx = candidates[0]
            else:
                # Multiple candidates: choose outermost (leftmost turn)
                # From LoopExtractor::findOutermost logic
                prev_dir = vec_angle_to(
                    edges[loop_indices[-1]]['start'][0],
                    edges[loop_indices[-1]]['start'][1],
                    current_end[0], current_end[1]
                )
                best_angle_diff = float('inf')
                next_idx = candidates[0]
                for c_idx in candidates:
                    ce = edges[c_idx]
                    # Try both directions of the candidate edge
                    for sp, ep in [(ce['start'], ce['end']), (ce['end'], ce['start'])]:
                        if vec_distance(sp[0], sp[1], current_end[0], current_end[1]) > ENDPOINT_TOLERANCE:
                            continue
                        cand_dir = vec_angle_to(current_end[0], current_end[1], ep[0], ep[1])
                        # Left turn = positive angle difference (CCW)
                        diff = get_angle_difference(prev_dir, cand_dir)
                        if diff < best_angle_diff:
                            best_angle_diff = diff
                            next_idx = c_idx

            processed[next_idx] = True
            ne = edges[next_idx]

            # Determine direction: does the edge's start or end connect to current_end?
            if vec_distance(ne['start'][0], ne['start'][1], current_end[0], current_end[1]) <= ENDPOINT_TOLERANCE:
                current_end = ne['end']
            else:
                current_end = ne['start']

            loop_indices.append(next_idx)

        # Validate: check closure
        if vec_distance(current_end[0], current_end[1], target[0], target[1]) <= ENDPOINT_TOLERANCE:
            # Build vertex list from edges
            vertices = []
            for idx in loop_indices:
                e = edges[idx]
                s = e['start']
                # Determine correct direction
                if not vertices:
                    vertices.append(s)
                end = e['end']
                # Check if we need to reverse
                if vec_distance(end[0], end[1], vertices[-1][0], vertices[-1][1]) < ENDPOINT_TOLERANCE:
                    vertices.append(e['start'])
                else:
                    vertices.append(end)

            # Ensure CCW orientation (positive area)
            if polygon_area_signed(vertices) < 0:
                vertices.reverse()

            # Check non-degenerate
            if polygon_area(vertices) > ENDPOINT_TOLERANCE * ENDPOINT_TOLERANCE:
                results.append(vertices)

        # Update unprocessed list
        unprocessed = [i for i, p in enumerate(processed) if not p]

    return results


# ============================================================
# 12. LOOP SORTER (hierarchical containment, from lc_looputils)
# ============================================================

def sort_loops_by_containment(loops: List[List[Tuple[float, float]]]) -> List[dict]:
    """Sort loops into a containment hierarchy.
    Ported from LC_LoopUtils::LoopSorter.

    Returns list of dicts: {
        'vertices': [...],
        'area': float,
        'children': [same structure, recursively]
    }
    Outer loops contain inner loops (holes). Smaller loops are children of larger ones.
    """
    if not loops:
        return []

    # Compute areas and bounding boxes
    loop_data = []
    for i, loop in enumerate(loops):
        area = polygon_area(loop)
        xs = [v[0] for v in loop]
        ys = [v[1] for v in loop]
        loop_data.append({
            'index': i,
            'vertices': loop,
            'area': area,
            'bbox': (min(xs), min(ys), max(xs), max(ys)),
            'children': []
        })

    # Sort by ascending absolute area (smallest first)
    loop_data.sort(key=lambda ld: ld['area'])

    # Build hierarchy: for each loop (small to large), find its parent
    # A parent is the smallest loop that contains this loop's centroid
    for ld in loop_data:
        cx, cy = polygon_centroid(ld['vertices'])
        parent = None
        for candidate in loop_data:
            if candidate['index'] == ld['index']:
                continue
            if candidate['area'] <= ld['area']:
                continue
            # Check if centroid is inside candidate
            if point_in_polygon_ray(cx, cy, candidate['vertices']):
                if parent is None or candidate['area'] < parent['area']:
                    parent = candidate
        if parent is not None:
            parent['children'].append(ld)

    # Return only root loops (those with no parent)
    roots = [ld for ld in loop_data if not any(
        ld['index'] in [c['index'] for c in other['children']]
        for other in loop_data
        if other['index'] != ld['index']
    )]

    # Better: identify roots as loops that are not children of any other loop
    child_indices = set()
    for ld in loop_data:
        for c in ld['children']:
            child_indices.add(c['index'])

    roots = [ld for ld in loop_data if ld['index'] not in child_indices]

    # Clean up internal data
    def _clean(ld):
        return {
            'vertices': ld['vertices'],
            'area': ld['area'],
            'children': [_clean(c) for c in ld['children']]
        }

    return [_clean(r) for r in roots]


def hierarchical_area(loop_tree: List[dict]) -> float:
    """Compute net area of a loop hierarchy (outer - holes + islands).
    Ported from LC_LoopUtils::LC_Loops::getTotalArea."""
    total = 0.0
    for node in loop_tree:
        area = node['area']
        child_area = hierarchical_area(node['children'])
        total += area - child_area
    return total


# ============================================================
# 13. ELLIPSE PARAMETRICS (from lc_looputils e_point, e_prime)
# ============================================================

def ellipse_point(cx, cy, major, minor, rotation, t):
    """Parametric point on ellipse.
    From LC_LoopUtils::LC_Loops::e_point.
    @param cx, cy: center
    @param major: semi-major axis length
    @param minor: semi-minor axis length
    @param rotation: rotation angle of major axis (radians)
    @param t: parameter (radians, 0 to 2*PI for full ellipse)
    @return (x, y) point on ellipse"""
    cos_t = math.cos(t)
    sin_t = math.sin(t)
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    # Point in local frame
    lx = major * cos_t
    ly = minor * sin_t
    # Rotate to world frame
    x = cx + lx * cos_r - ly * sin_r
    y = cy + lx * sin_r + ly * cos_r
    return (x, y)


def ellipse_tangent(major, minor, rotation, t):
    """Tangent vector (dx/dt, dy/dt) at parameter t on ellipse.
    From LC_LoopUtils::LC_Loops::e_prime."""
    cos_t = math.cos(t)
    sin_t = math.sin(t)
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    # Derivative in local frame
    dlx = -major * sin_t
    dly = minor * cos_t
    # Rotate to world frame
    dx = dlx * cos_r - dly * sin_r
    dy = dlx * sin_r + dly * cos_r
    return (dx, dy)


def ellipse_arc_to_line_segments(cx, cy, major, minor, rotation, a1, a2,
                                  num_segments: int = 16) -> List[Tuple[float, float]]:
    """Approximate an elliptic arc with line segments.
    From LC_LoopUtils::addEllipticArc logic (which uses cubic Beziers in C++).
    For M2, line segment approximation is sufficient.
    @param a1, a2: start and end parameters (radians)
    @param num_segments: number of line segments for approximation
    @return: list of (x, y) points along the arc"""
    if abs(a2 - a1) < RS_TOLERANCE_ANGLE:
        return []
    points = []
    for i in range(num_segments + 1):
        t = a1 + (a2 - a1) * i / num_segments
        points.append(ellipse_point(cx, cy, major, minor, rotation, t))
    return points


# ============================================================
# 14. POLYGON UTILITIES
# ============================================================

def polygon_orientation(vertices: List[Tuple[float, float]]) -> str:
    """Determine polygon winding: 'CCW' or 'CW'.
    Based on signed area sign from shoelace formula."""
    sa = polygon_area_signed(vertices)
    return "CCW" if sa > 0 else "CW" if sa < 0 else "DEGENERATE"


def ensure_ccw(vertices: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Ensure polygon is counter-clockwise. Reverse if needed.
    From LoopExtractor's area-based orientation correction."""
    if polygon_area_signed(vertices) < 0:
        return list(reversed(vertices))
    return list(vertices)


def polygon_bounding_box(vertices: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding box. Returns (xmin, ymin, xmax, ymax)."""
    if not vertices:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def polygon_perimeter(vertices: List[Tuple[float, float]]) -> float:
    """Total perimeter length of a polygon."""
    n = len(vertices)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        total += vec_distance(vertices[i][0], vertices[i][1],
                              vertices[j][0], vertices[j][1])
    return total


def douglas_peucker(points: List[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
    """Douglas-Peucker line simplification algorithm.
    Useful for simplifying parcel polygon boundaries extracted from raster tiles."""
    if len(points) <= 2:
        return list(points)

    # Find the point with maximum distance from line(start, end)
    sx, sy = points[0]
    ex, ey = points[-1]
    max_dist = 0.0
    max_idx = 0

    for i in range(1, len(points) - 1):
        d = parallel_line_distance(sx, sy, ex, ey, points[i][0], points[i][1])[0]
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > tolerance:
        left = douglas_peucker(points[:max_idx + 1], tolerance)
        right = douglas_peucker(points[max_idx:], tolerance)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def polygon_has_self_intersection(vertices: List[Tuple[float, float]]) -> bool:
    """Check if a polygon has any self-intersecting edges.
    O(n^2) brute force — adequate for parcel polygons (< 100 vertices).
    Returns True if any non-adjacent edges cross."""
    n = len(vertices)
    if n < 4:
        return False
    for i in range(n):
        j = (i + 1) % n
        for k in range(i + 2, n):
            l = (k + 1) % n
            # Skip adjacent edges
            if j == k or i == l:
                continue
            # Check if segments (i,j) and (k,l) intersect
            pt = line_line_intersection(vertices[i], vertices[j], vertices[k], vertices[l])
            if pt is not None:
                return True
    return False


def polygon_simplify_by_angle(vertices: List[Tuple[float, float]], angle_tol_deg: float = 5.0) -> List[Tuple[float, float]]:
    """Remove vertices where the interior angle is close to 180 degrees (nearly collinear).
    From LC_LineMath::angleFor3Points concept."""
    if len(vertices) <= 3:
        return list(vertices)
    angle_tol = deg2rad(angle_tol_deg)
    result = []
    n = len(vertices)
    for i in range(n):
        p = vertices[(i - 1) % n]
        c = vertices[i]
        q = vertices[(i + 1) % n]
        angle = angle_for_3_points(p, c, q)
        if angle > angle_tol:
            result.append(c)
    return result


def polygons_overlap(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> bool:
    """Check if two polygons overlap (any edge of one crosses any edge of the other,
    or one contains the other's vertex). From LC_LoopUtils::LC_Loops::overlap logic."""
    # Quick bbox check
    bb1 = polygon_bounding_box(poly1)
    bb2 = polygon_bounding_box(poly2)
    if bb1[2] < bb2[0] or bb1[0] > bb2[2] or bb1[3] < bb2[1] or bb1[1] > bb2[3]:
        return False
    # Check if any vertex of poly1 is inside poly2
    for v in poly1:
        if point_in_polygon_ray(v[0], v[1], poly2):
            return True
    # Check if any vertex of poly2 is inside poly1
    for v in poly2:
        if point_in_polygon_ray(v[0], v[1], poly1):
            return True
    # Check edge crossings
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        j = (i + 1) % n1
        for k in range(n2):
            l = (k + 1) % n2
            if line_line_intersection(poly1[i], poly1[j], poly2[k], poly2[l]) is not None:
                return True
    return False


# ============================================================
# 15. DXF GROUP CODES (from rs_filterdxf1)
# ============================================================

DXF_GROUP_CODES = {
    0: "Entity type",
    1: "Primary text value",
    2: "Name / block name / attribute tag",
    8: "Layer name",
    10: "Primary X coordinate (point / line start / circle center)",
    11: "Primary Y coordinate",
    12: "Primary Z coordinate",
    20: "Secondary X (text alignment / etc.)",
    21: "Secondary Y",
    30: "Tertiary X",
    31: "Tertiary Y",
    40: "Radius / text height / scale factor",
    50: "Angle in degrees",
    60: "Entity visibility (0=visible, 1=hidden)",
    62: "Color number",
    70: "Flag / closed polyline flag (1=closed)",
    71: "Text generation flags / vertex count",
    90: "Number of vertices (LWPOLYLINE)",
    100: "Subclass marker",
}

DXF_ENTITY_TYPE_STRINGS = {
    "POINT": DXF_ENTITY_POINT,
    "LINE": DXF_ENTITY_LINE,
    "LWPOLYLINE": DXF_ENTITY_LWPOLYLINE,
    "POLYLINE": DXF_ENTITY_POLYLINE,
    "ARC": DXF_ENTITY_ARC,
    "CIRCLE": DXF_ENTITY_CIRCLE,
    "ELLIPSE": DXF_ENTITY_ELLIPSE,
    "INSERT": DXF_ENTITY_INSERT,
    "TEXT": DXF_ENTITY_TEXT,
    "MTEXT": DXF_ENTITY_MTEXT,
    "SOLID": DXF_ENTITY_SOLID,
}

DXF_LWPOLYLINE_CODES = {
    90: "Number of vertices",
    70: "LWPOLYLINE flag (1=closed, 128=Plinegen)",
    43: "Constant width",
    40: "Bulge (arc tangent for arc segments)",
    42: "Wiggle (dash pattern for old-style)",
}

DXF_MTEXT_CODES = {
    71: "Text attachment point (1=top-left, 2=top-center, 3=top-right, "
         "4=middle-left, 5=middle-center, 6=middle-right, 7=bottom-left, "
         "8=bottom-center, 9=bottom-right)",
    10: "Insertion point X",
    20: "Insertion point Y",
    40: "Nominal text height",
    41: "Reference rectangle width",
    42: "Reference rectangle height",
    3: "MTEXT content (additional text, appended to group 1)",
    50: "Column spacing",
}

DXF_LINE_CODES = {
    10: "Start X", 20: "Start Y",
    11: "End X",   21: "End Y",
}

DXF_ARC_CODES = {
    10: "Center X", 20: "Center Y",
    40: "Radius",
    50: "Start angle (degrees)",
    51: "End angle (degrees)",
}

DXF_CIRCLE_CODES = {
    10: "Center X", 20: "Center Y",
    40: "Radius",
}

DXF_ELLIPSE_CODES = {
    10: "Center X", 20: "Center Y",
    11: "Major axis endpoint X (relative to center)", 21: "Major axis endpoint Y",
    40: "Minor axis ratio (minor/major)",
    41: "Start parameter",
    42: "End parameter (2*PI - start for full ellipse)",
}

DXF_POINT_CODES = {
    10: "Point X", 20: "Point Y", 30: "Point Z (if 3D)",
}


# ============================================================
# 16. BULGE TO ARC CONVERSION (DXF LWPOLYLINE)
# ============================================================

def bulge_to_arc(p1x, p1y, p2x, p2y, bulge: float):
    """Convert a bulge arc segment to arc parameters. From LibreCAD DXF handling.
    Bulge = tan(included_angle/4). Returns (cx, cy, r, start_angle, end_angle) or None."""
    if abs(bulge) < RS_TOLERANCE:
        return None
    dx = p2x - p1x
    dy = p2y - p1y
    d = math.hypot(dx, dy)
    if d < RS_TOLERANCE:
        return None
    alpha = 2.0 * math.atan(abs(bulge))
    if alpha < RS_TOLERANCE:
        return None
    sagitta = d / 2.0 * abs(bulge)
    r = sagitta / 2.0 + d * d / (8.0 * sagitta)
    if r < RS_TOLERANCE:
        return None
    mx, my = (p1x + p2x) / 2, (p1y + p2y) / 2
    perp_dist = r - sagitta
    chord_angle = math.atan2(dy, dx)
    if bulge > 0:
        perp_angle = chord_angle + math.pi / 2
    else:
        perp_angle = chord_angle - math.pi / 2
    cx = mx + perp_dist * math.cos(perp_angle)
    cy = my + perp_dist * math.sin(perp_angle)
    sa = math.atan2(p1y - cy, p1x - cx)
    ea = math.atan2(p2y - cy, p2x - cx)
    return (cx, cy, r, sa, ea)


def bulge_to_points(p1x, p1y, p2x, p2y, bulge: float, num_points: int = 8) -> List[Tuple[float, float]]:
    """Convert a bulge arc segment to a series of line-approximation points.
    Returns list of points from p1 to p2 along the arc."""
    arc = bulge_to_arc(p1x, p1y, p2x, p2y, bulge)
    if arc is None:
        return [(p1x, p1y), (p2x, p2y)]
    cx, cy, r, sa, ea = arc
    # Determine sweep direction
    if bulge > 0:
        if ea <= sa:
            ea += 2.0 * math.pi
        sweep = ea - sa
    else:
        if sa <= ea:
            sa += 2.0 * math.pi
        sweep = sa - ea
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        if bulge > 0:
            angle = sa + sweep * t
        else:
            angle = ea + sweep * t
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return points


# ============================================================
# 17. SECOND MOMENTS OF AREA (from lc_looputils / lc_secondmoment)
# ============================================================

def polygon_second_moments(vertices):
    """Compute Ixx, Iyy, Ixy (second moments of area). From LC_SecondMoment.
    Uses Green's theorem formulation.
    Ixx = integral(y^2 dx), Iyy = integral(x^2 dx), Ixy = integral(xy dx)"""
    n = len(vertices)
    if n < 3:
        return (0.0, 0.0, 0.0)
    ixx = iyy = ixy = 0.0
    for i in range(n):
        j = (i + 1) % n
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        cross = xi * yj - xj * yi
        ixx += (yi * yj + yi * yj + yi * yi) * (xi * xi + xi * xj + xj * xj)
        iyy -= (xi * xj + xi * xj + xj * xj) * (yi * yi + yi * yj + yj * yj)
        ixy += (xi * xj * (2 * yj + yi) + xi * xi * (2 * yj + yi)) * cross
    ixx /= 12.0
    iyy /= 12.0
    ixy /= 24.0
    return (ixx, iyy, ixy)


# ============================================================
# 18. DOUBLE TO STRING (from rs_math)
# ============================================================

def double_to_string(value: float, prec: float) -> str:
    """Convert a double to the shortest string that rounds back to the same value
    at the given precision. From RS_Math::doubleToString(double, double)."""
    if prec < RS_TOLERANCE:
        return str(value)
    num = round(value / prec) * prec
    # Determine digits after decimal point
    exa_str = f"{1.0 / prec:.10f}"
    dot_pos = exa_str.find('.')
    if dot_pos == -1:
        return str(round(num))
    digits = dot_pos  # number of digits after decimal
    result = f"{num:.{digits}f}"
    # Remove trailing zeros after decimal
    if '.' in result:
        result = result.rstrip('0').rstrip('.')
    return result


# ============================================================
# M2 INTEGRATION HELPERS
# ============================================================

def rigid_transform_points(points, angle_rad, scale, tx, ty):
    """Apply rigid transformation (rotation + scale + translation) to a list of points.
    This is the core operation for FMB georeferencing in M2.
    @param points: list of (x, y)
    @param angle_rad: rotation angle
    @param scale: scale factor
    @param tx, ty: translation
    @return: list of transformed (x, y)"""
    c = math.cos(angle_rad) * scale
    s = math.sin(angle_rad) * scale
    return [(p[0] * c - p[1] * s + tx, p[0] * s + p[1] * c + ty) for p in points]


def compute_rigid_transform(src_points, dst_points):
    """Compute rigid transformation (rotation + scale + translation) from src to dst.
    Uses SVD-based approach (Kabsch algorithm variant with scale).
    @param src_points: list of (x, y) source points
    @param dst_points: list of (x, y) destination points
    @return: (angle_rad, scale, tx, ty) or None if degenerate
    """
    n = len(src_points)
    if n < 2 or n != len(dst_points):
        return None

    # Centroids
    sx = sum(p[0] for p in src_points) / n
    sy = sum(p[1] for p in src_points) / n
    dx = sum(p[0] for p in dst_points) / n
    dy = sum(p[1] for p in dst_points) / n

    # Centered points
    sc = [(p[0] - sx, p[1] - sy) for p in src_points]
    dc = [(p[0] - dx, p[1] - dy) for p in dst_points]

    # Compute scale
    src_var = sum(p[0] ** 2 + p[1] ** 2 for p in sc)
    if src_var < RS_TOLERANCE2:
        return None

    # Cross-covariance
    cov_xx = sum(sc[i][0] * dc[i][0] for i in range(n))
    cov_xy = sum(sc[i][0] * dc[i][1] for i in range(n))
    cov_yx = sum(sc[i][1] * dc[i][0] for i in range(n))
    cov_yy = sum(sc[i][1] * dc[i][1] for i in range(n))

    # SVD of 2x2 covariance matrix (analytical)
    # C = [[cov_xx, cov_xy], [cov_yx, cov_yy]]
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_yx
    discriminant = max(0.0, trace * trace / 4.0 - det)
    s1_sq = trace / 2.0 + math.sqrt(discriminant)
    s2_sq = trace / 2.0 - math.sqrt(discriminant)
    s1 = math.sqrt(max(0.0, s1_sq))
    s2 = math.sqrt(max(0.0, s2_sq))

    if s1 < RS_TOLERANCE:
        return None

    scale = (s1 + s2) / src_var if src_var > 0 else 1.0

    # Rotation angle
    angle = math.atan2(cov_yx - cov_xy, cov_xx + cov_yy)

    # Translation
    tx = dx - scale * (sx * math.cos(angle) - sy * math.sin(angle))
    ty = dy - scale * (sx * math.sin(angle) + sy * math.cos(angle))

    return (angle, scale, tx, ty)


def compute_residuals(src_points, dst_points, angle_rad, scale, tx, ty):
    """Compute per-point residuals after rigid transformation.
    @return: list of residual distances in meters"""
    transformed = rigid_transform_points(src_points, angle_rad, scale, tx, ty)
    return [vec_distance(t[0], t[1], d[0], d[1])
            for t, d in zip(transformed, dst_points)]


def iou_polygons(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> float:
    """Compute IoU (Intersection over Union) of two simple polygons.
    Uses the shoelace-based signed area approach for convex polygons.
    For general polygons, use shapely (this is a fallback)."""
    try:
        from shapely.geometry import Polygon
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)
        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        if union < RS_TOLERANCE:
            return 0.0
        return inter / union
    except ImportError:
        # Fallback: approximate with convex hull IoU
        h1 = convex_hull(poly1)
        h2 = convex_hull(poly2)
        a1 = polygon_area(h1)
        a2 = polygon_area(h2)
        # For convex polygons, approximate intersection
        # This is a rough approximation — use shapely for accuracy
        return min(a1, a2) / max(a1, a2) if max(a1, a2) > 0 else 0.0