"""Place an M1 FMB plot using the S3 cadastre ONLY as a position+rotation reference.

ARCHITECTURE (hard requirement): the S3 cadastral tiles are a REFERENCE, never a
geometry source. From the S3 parcel we take ONLY:
  * position  -- where the plot sits in UTM (parcel centroid / OCR label point),
  * rotation  -- the plot's real-world orientation (best rigid alignment angle).
The plot GEOMETRY is the M1 FMB (from the FMB PDF -> DXF). We apply a RIGID
similarity (rotation + uniform scale ~1 + translation) to the M1 geometry and place
it -- the FMB shape is preserved EXACTLY. We do NOT:
  * wrap the S3 polygon as fake surveyor stones,
  * pin M1 corners onto S3 vertices (cadastral_adjust) -- that warps the clean FMB
    onto the noisy z18 raster boundary,
  * gate on IoU-against-the-S3-pixel-boundary.

Quality is judged by AREA RATIO (M1 plot area vs S3 parcel area) -- a size/identity
check that never deforms geometry -- plus the caller's village/corridor + non-overlap
gates. Real field control (surveyor stones) is applied separately by the caller as a
rigid re-fit; that is the authoritative path and is left untouched.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from ..m2_georef.extract_m1 import M1PlotData
from ..m2_georef.transform import umeyama
from .source import CadastralParcel

_log = logging.getLogger(__name__)


@dataclass
class CadastralFit:
    adjusted: np.ndarray          # (N,2) UTM positions for ALL M1 stones (rigid)
    R: np.ndarray
    s: float
    t: np.ndarray
    method: str                   # "rigid" | "anchor_rotated" | "anchor"
    n_inliers: int
    area_ratio: float             # placed M1 area / parcel area (quality, NOT IoU)
    rot_residual: float = float("inf")   # corner-alignment residual (m), diagnostic
    orientation_ok: bool = True   # rotation aligns M1<->parcel principal axes (A2 flip gate)
    rot_residual_robust: float = float("inf")  # trimmed-mean variant of rot_residual (see below)


def _principal_angle(coords: np.ndarray) -> float:
    """Angle (radians) of the first principal component of a point set."""
    centered = coords - coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return math.atan2(Vt[0, 1], Vt[0, 0])


def _is_near_square(coords: np.ndarray, tolerance: float = 0.30) -> bool:
    """True if the point cloud is ~square (singular-value aspect ~1). For such a
    plot the principal axis is unstable and orientation cannot be validated."""
    centered = coords - coords.mean(axis=0)
    s = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    if len(s) < 2:
        return True
    aspect = s[0] / max(s[1], 1e-9)
    return aspect < 1.0 + tolerance


def _orientation_consistent(R: np.ndarray, parcel: CadastralParcel,
                            m1_ring: np.ndarray) -> bool:
    """A2 flip gate: the rigid rotation R must align the M1 principal axis to within
    45 degrees of the parcel principal axis. Rejects gross (~90 degrees) flips that
    ICP can settle into on a near-symmetric plot. Near-square plots have no stable
    axis, so they pass (the area/scale gates carry them instead).
    """
    if parcel.polygon is None or parcel.polygon.area < 10.0:
        return True
    if _is_near_square(m1_ring):
        return True
    par_coords = np.array(list(parcel.polygon.exterior.coords)[:-1])
    par_angle = _principal_angle(par_coords)
    m1_angle = _principal_angle(m1_ring)
    rot_angle = math.atan2(R[1, 0], R[0, 0])
    diff = (m1_angle + rot_angle - par_angle) % math.pi
    if diff > math.pi / 2:
        diff = math.pi - diff
    return diff < math.pi / 4


def _poly_area(ring: np.ndarray) -> float:
    try:
        p = Polygon([(float(x), float(y)) for x, y in ring])
        if not p.is_valid:
            p = p.buffer(0)
        return float(p.area)
    except Exception:  # noqa: BLE001
        return 0.0


def _skeleton_corners(poly, target_n: int) -> np.ndarray | None:
    """Reduce a (noisy, many-vertex) parcel outline to ~target_n MAJOR corners via
    Douglas-Peucker at increasing tolerance -- the straight-edge skeleton. Used ONLY
    to estimate the rotation angle; the parcel geometry is never output."""
    try:
        coords0 = np.array(poly.exterior.coords[:-1])
    except Exception:  # noqa: BLE001
        return None
    best = coords0
    for tol in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0):
        s = poly.simplify(tol, preserve_topology=True)
        if s.is_empty or s.geom_type != "Polygon":
            continue
        c = np.array(s.exterior.coords[:-1])
        best = c
        if len(c) <= target_n:
            break
    return best if len(best) >= 3 else None


def _densify_ring(coords, step: float = 3.0) -> np.ndarray:
    """Points along a polygon boundary every ~step metres (the ICP target cloud)."""
    out = []
    n = len(coords)
    for i in range(n):
        a = np.array(coords[i], float)
        b = np.array(coords[(i + 1) % n], float)
        out.append(a)
        d = float(np.hypot(*(b - a)))
        for k in range(1, int(d // step)):
            out.append(a + (b - a) * (k * step / d))
    return np.array(out) if out else np.array(coords, float)


def _icp_rigid(src_ring: np.ndarray, target_pts: np.ndarray, init, iters: int = 12):
    """2D rigid ICP: align src_ring (M1 corners) to target_pts (parcel boundary)
    by iterated nearest-neighbour + Umeyama. Rigid only -- never deforms src."""
    from scipy.spatial import cKDTree
    tree = cKDTree(target_pts)
    R, s, t = init
    for _ in range(iters):
        cur = s * (src_ring @ R.T) + t
        _, idx = tree.query(cur)
        R2, s2, t2 = umeyama(src_ring, target_pts[idx])[:3]
        if not (0.5 < s2 < 2.0):
            break
        if np.allclose(R2, R, atol=1e-6) and abs(s2 - s) < 1e-6:
            R, s, t = R2, s2, t2
            break
        R, s, t = R2, s2, t2
    cur = s * (src_ring @ R.T) + t
    d, _ = tree.query(cur)
    return R, s, t, float(d.mean())


# Fraction of the closest corner->boundary distances kept by the robust residual.
# 0.8 drops the worst ~1 corner on a typical 4-6 corner ring -- enough to shrug off a
# single OCR/raster-jittered corner, not so aggressive it hides a genuinely bad fit.
_ROBUST_RESID_KEEP = 0.8


def _robust_corner_residual(placed_ring: np.ndarray, target_pts: np.ndarray,
                            keep: float = _ROBUST_RESID_KEEP) -> float:
    """Trimmed-mean nearest-neighbour distance from the placed M1 corners to the parcel
    boundary cloud -- a partial-Chamfer / Modified-Hausdorff variant of ``rot_residual``.

    Adopted from the M2 open-source "diamonds" review. Rationale: our plain residual is
    already a MEAN nearest-neighbour distance (not brittle Hausdorff), but the raster-traced
    parcel boundary has noisy corners, so a single jittered vertex can inflate the mean and
    fail an otherwise-correct fit ONLY on residual. Trimming the worst ``1-keep`` fraction
    makes the fit-quality estimate robust to that, recovering recall.

    One-directional (corners -> dense boundary) BY DESIGN: the target is a densified boundary
    cloud with far more points than the ring, so a symmetric chamfer would be swamped by the
    point-count asymmetry. Equals the plain mean on a clean fit (nothing to trim)."""
    from scipy.spatial import cKDTree
    placed = np.asarray(placed_ring, float)
    if placed.size == 0 or target_pts.size == 0:
        return float("inf")
    d, _ = cKDTree(target_pts).query(placed)
    d = np.sort(np.atleast_1d(d))
    k = max(3, int(round(len(d) * keep)))
    return float(d[:k].mean())


def _placed_iou(m1_ring: np.ndarray, R, s, t, parcel_poly) -> float:
    """Footprint IoU of the rigidly-placed M1 corner ring against the parcel polygon.

    This is a ROTATION-DISAMBIGUATION signal, not a geometry source: corner residual
    alone is rotation-ambiguous on a near-square plot/parcel (several orientations have
    near-equal corner residual), so the lowest-residual pose can be ~90 deg off the one
    that actually FILLS the parcel. IoU measures fill/overlap and cleanly separates them.
    Returns 0.0 on any degenerate geometry.
    """
    if parcel_poly is None:
        return 0.0
    try:
        placed = s * (m1_ring @ R.T) + t
        p = Polygon([(float(x), float(y)) for x, y in placed])
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area <= 0:
            return 0.0
        inter = p.intersection(parcel_poly).area
        union = p.area + parcel_poly.area - inter
        return float(inter / union) if union > 0 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


# Tie band for the IoU-disambiguated selection: among ICP-refined candidates whose
# corner residual is within RESID_TIE_FACTOR * (best residual + floor), prefer the one
# with the highest footprint IoU against the parcel. The floor avoids a knife-edge band
# when the best residual is tiny. This is GENERAL geometry -- a well-fit plot's
# best-residual pose already has the best IoU, so it is unchanged; only a rotation-
# ambiguous near-square plot, where a wrong-rotation pose sneaks a marginally lower
# residual, is corrected (the well-filling pose wins).
_RESID_TIE_FACTOR = 1.35
_RESID_TIE_FLOOR_M = 2.0


def _rigid_from_parcel(m1_ring: np.ndarray, parcel: CadastralParcel):
    """Best RIGID (rotation+scale+translation) aligning the M1 corner ring to the
    parcel boundary -- a cyclic-Procrustes initial guess refined by 2D ICP. Rigid
    throughout, so it recovers the real-world ORIENTATION+POSITION from the S3
    reference without deforming the FMB. Returns (R, s, t, residual) or None.

    Candidate selection: corner residual is rotation-ambiguous on near-square plots,
    so we ICP-refine the top-scoring cyclic seeds and, among those whose residual is
    within a small factor of the best, pick the one with the highest footprint IoU
    against the parcel (the pose that best FILLS the parcel). A well-fit plot's
    best-residual pose already has the best IoU, so this is a no-op there.
    """
    n = len(m1_ring)
    par = _skeleton_corners(parcel.polygon, n)
    if par is None:
        return None
    k = min(n, len(par))
    if k < 3:
        return None
    # Initial guesses: cyclic rotation/reflection vs the parcel skeleton corners.
    # Keep the best-residual seed PER DISTINCT ROTATION ANGLE (rounded), so refinement
    # explores all candidate orientations -- not just many shifts of the single best.
    seeds: dict[int, tuple[float, np.ndarray, float, np.ndarray]] = {}
    for seq in (m1_ring, m1_ring[::-1]):
        for om in range(n):
            sm = np.roll(seq, -om, axis=0)[:k]
            for op in range(len(par)):
                pm = np.roll(par, -op, axis=0)[:k]
                R, s, t, res = umeyama(sm, pm)
                if not (0.5 < s < 2.0):
                    continue
                r = float(res.mean())
                ang = int(round(math.degrees(math.atan2(R[1, 0], R[0, 0])))) % 360
                prev = seeds.get(ang)
                if prev is None or r < prev[0]:
                    seeds[ang] = (r, R, s, t)
    if not seeds:
        return None

    target = _densify_ring(list(parcel.polygon.exterior.coords)[:-1])
    parcel_poly = parcel.polygon

    # ICP-refine each distinct-orientation seed; record (residual, IoU, R, s, t).
    refined: list[tuple[float, float, np.ndarray, float, np.ndarray]] = []
    for _r0, R0, s0, t0 in seeds.values():
        R, s, t, resid = _icp_rigid(m1_ring, target, (R0, s0, t0))
        if not (0.5 < s < 2.0):
            continue
        iou = _placed_iou(m1_ring, R, s, t, parcel_poly)
        refined.append((resid, iou, R, s, t))
    if not refined:
        return None

    best_resid = min(c[0] for c in refined)
    band = _RESID_TIE_FACTOR * (best_resid + _RESID_TIE_FLOOR_M)
    # Among candidates whose residual is within the tie band of the best, pick max IoU.
    # Tie-break on lower residual. Outside the band, the lowest-residual pose still wins.
    in_band = [c for c in refined if c[0] <= band]
    pool = in_band if in_band else refined
    resid, _iou, R, s, t = max(pool, key=lambda c: (c[1], -c[0]))
    return R, s, t, resid


def fit_plot_to_parcel(
    m1: M1PlotData,
    parcel: CadastralParcel,
    anchor: tuple[float, float] | None = None,
) -> CadastralFit | None:
    """Rigidly place the M1 FMB using the S3 parcel as a position+rotation reference."""
    corners = m1.outer_stone_indices
    if len(corners) < 3:
        return None
    m1_pos = m1.stone_positions()
    ring = m1_pos[np.array(corners)]

    # --- preferred: derive rotation+position from the parcel by rigid alignment ---
    if parcel.polygon is not None and parcel.polygon.area > 1.0:
        fit = _rigid_from_parcel(ring, parcel)
        if fit is not None:
            R, s, t, resid = fit
            adjusted = s * (m1_pos @ R.T) + t          # RIGID -> M1 shape preserved
            placed = adjusted[np.array(corners)]
            ar = _poly_area(placed) / max(parcel.polygon.area, 1e-9)
            ok = _orientation_consistent(R, parcel, ring)
            robust = _robust_corner_residual(
                placed, _densify_ring(list(parcel.polygon.exterior.coords)[:-1]))
            return CadastralFit(adjusted=adjusted, R=R, s=s, t=t, method="rigid",
                                n_inliers=len(corners), area_ratio=ar,
                                rot_residual=resid, orientation_ok=ok,
                                rot_residual_robust=robust)

    # --- fallback: position (centroid/label) + principal-axis rotation ---
    tgt = anchor
    if tgt is None and parcel.polygon is not None:
        c = parcel.polygon.centroid
        tgt = (c.x, c.y)
    if tgt is None:
        return None
    ring_c = ring.mean(axis=0)
    if parcel.polygon is not None:
        par_coords = np.array(list(parcel.polygon.exterior.coords)[:-1])
        delta = _principal_angle(par_coords) - _principal_angle(ring)
        c_, s_ = math.cos(delta), math.sin(delta)
        R = np.array([[c_, -s_], [s_, c_]])
        adjusted = (m1_pos - ring_c) @ R.T + np.array(tgt)
        placed = adjusted[np.array(corners)]
        ar = _poly_area(placed) / max(parcel.polygon.area, 1e-9)
        return CadastralFit(adjusted=adjusted, R=R, s=1.0,
                            t=np.array(tgt) - ring_c @ R.T, method="anchor_rotated",
                            n_inliers=0, area_ratio=ar)
    t = np.array([tgt[0] - ring_c[0], tgt[1] - ring_c[1]])
    return CadastralFit(adjusted=m1_pos + t, R=np.eye(2), s=1.0, t=t,
                        method="anchor", n_inliers=0, area_ratio=0.0)
