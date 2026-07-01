"""Stone-snap refine: pull each clubbed FMB onto the REAL surveyor stones (0-FP).

The S3/TNGIS cadastre seats a plot to ~10-15 m. When the surveyor's actual boundary
stones are available (the client's field data / "compared" reference), we can do what the
manual workers do: rigidly match each FMB's corner stones onto the true stones and snap
the plot there. That is the ONLY route to <2 m, manual-quality placement + merged
boundaries (the S3 cadastre alone cannot reach it).

DISCIPLINE (0-FALSE-POSITIVE, RIGID)
------------------------------------
* The refine is a RIGID similarity (rotation + uniform scale ~1 + translation) fitted by
  RANSAC over corner<->stone correspondences -- shape is NEVER warped.
* A plot is refined ONLY on a CONFIDENT congruent fit: enough inlier corners land on
  distinct true stones within ``tol`` at low residual, scale stays ~1, AND the correction
  is small enough to be a refine of the cadastre seat (not a jump to a look-alike parcel
  elsewhere -- ``max_shift``). Otherwise the cadastre seat is KEPT and the plot flagged.
* Refined plots are marked ``anchored`` so the downstream edge-align/​corner-snap treat
  them as fixed truth (unrefined neighbours snap TO them, never the reverse).

This is the automated FMBS_STONES_MATCH against authoritative field data; the deterministic
gate still decides, so it stays 0-FP.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .placement import ClubResult

_log = logging.getLogger(__name__)

# PRINCIPLED gates (pure physics/geometry -- identical on any village, any metre-scale):
TOL = 2.5              # m: a corner within this of a stone is "on it" (survey stone accuracy)
MAX_RESIDUAL = 1.5    # m: mean inlier residual must be below this (sub-2 m == real stones)
SCALE_BAND = (0.97, 1.03)  # a rigid stone match is ~unit scale
# A rigid 2D fit has 4 DOF, so >=4 DISTINCT corner<->stone correspondences over-determine it.
# ABSOLUTE count (not a fraction of corners) -- an FMB ring carries intermediate boundary
# vertices that are not surveyed stones, so a high fraction would reject a correct plot.
MIN_INLIERS = 4
MIN_INLIER_FRAC = 0.20
ICP_ITERS = 12        # ICP polish sweeps after a RANSAC seed
OVERLAP_MAX = 0.15    # reject a stone-snap that stacks the plot on another by > this frac
# SIZE-RELATIVE distances -- a FRACTION of the plot's OWN diagonal (floored), so NONE is a
# fixed metre value tuned to one village's plots/cadastre. Derived per plot in ``_best_fit``.
NEAR_FRAC = 1.5       # search stones within this * plot-diagonal of the centroid
NEAR_FLOOR = 60.0     # m
SHIFT_FRAC = 0.4      # a true refine correction is < this * plot-diagonal (else = mis-match)
SHIFT_FLOOR = 20.0    # m: floor so a small plot can still absorb the ~raster cadastre error
MINSEP_FRAC = 0.1     # a RANSAC seed corner-pair must span >= this * plot-diagonal
MINSEP_FLOOR = 8.0    # m


@dataclass
class RefineStats:
    n_plots: int = 0
    n_refined: int = 0
    anchored: set = field(default_factory=set)     # survey numbers refined to stones
    per_plot: dict = field(default_factory=dict)   # survey -> (inliers, residual, shift)
    skipped: list = field(default_factory=list)    # (survey, reason)


def _rigid_2pt(p1, p2, q1, q2):
    vp, vq = p2 - p1, q2 - q1
    lp = float(np.hypot(*vp))
    if lp < 1e-6:
        return None
    s = float(np.hypot(*vq)) / lp
    if not (SCALE_BAND[0] < s < SCALE_BAND[1]):
        return None
    th = np.arctan2(vq[1], vq[0]) - np.arctan2(vp[1], vp[0])
    c, sn = np.cos(th), np.sin(th)
    R = np.array([[c, -sn], [sn, c]])
    return R, (q1 - R @ p1), s


def _umeyama_rigid(P: np.ndarray, Q: np.ndarray):
    """Least-squares similarity (rotation + uniform scale, clamped ~1) mapping P->Q."""
    mp, mq = P.mean(0), Q.mean(0)
    Pc, Qc = P - mp, Q - mq
    H = Pc.T @ Qc / len(P)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    var = (Pc ** 2).sum() / len(P)
    s = float((S * np.array([1.0, d])).sum() / var) if var > 1e-9 else 1.0
    s = min(SCALE_BAND[1], max(SCALE_BAND[0], s))
    t = mq - s * R @ mp
    return s * R, t


def _icp(corners, tree, truth, R, t, tol):
    """Polish a seed (R,t) by iterated closest point over corner<->stone associations."""
    for _ in range(ICP_ITERS):
        proj = (R @ corners.T).T + t
        dist, idx = tree.query(proj)
        m = dist <= tol * 1.6                     # associate a bit loosely while converging
        if m.sum() < 3:
            break
        # one stone per corner; drop duplicate stone claims (keep the closest corner).
        pairs = {}
        for ci in np.where(m)[0]:
            si = int(idx[ci])
            if si not in pairs or dist[ci] < pairs[si][1]:
                pairs[si] = (ci, dist[ci])
        if len(pairs) < 3:
            break
        P = np.array([corners[ci] for si, (ci, _) in pairs.items()])
        Q = np.array([truth[si] for si in pairs])
        R2, t2 = _umeyama_rigid(P, Q)
        if np.allclose(R2, R, atol=1e-6) and np.allclose(t2, t, atol=1e-4):
            R, t = R2, t2
            break
        R, t = R2, t2
    return R, t


def _pca_frame(pts: np.ndarray):
    """Centroid + principal axes (rows of V) of a point cloud, via SVD."""
    c = pts.mean(0)
    _u, _s, Vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, Vt


def _pca_seeds(corners: np.ndarray, truth_local: np.ndarray):
    """Global-pose seeds aligning the FMB's principal axes to the local stones' axes.

    Robust starting poses that don't depend on guessing a correct corner<->stone pair --
    the idea borrowed from the ICP contour-comparer (written clean here). The 4 axis-sign
    combinations cover the orientation ambiguity; each is ICP-polished + gated downstream.
    """
    if len(truth_local) < 3:
        return []
    cc, Vc = _pca_frame(corners)
    ct, Vt = _pca_frame(truth_local)
    seeds = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            R = Vt.T @ np.diag([sx, sy]) @ Vc
            if np.linalg.det(R) <= 0:            # keep proper rotations (no reflection)
                continue
            seeds.append((R, ct - R @ cc))
    return seeds


def _diag(pts: np.ndarray) -> float:
    """Plot size = bounding-box diagonal of its corner ring (metre-scale-independent)."""
    if len(pts) < 2:
        return 0.0
    return float(np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))


def _best_fit(corners: np.ndarray, truth: np.ndarray, tol: float):
    """Best rigid (R, t) mapping corners -> truth stones, (RANSAC + PCA) seed + ICP polish.

    All distance limits derive from the plot's OWN diagonal (size-relative, floored), so
    nothing is tuned to a village's metre-scale. Returns (R,t,inliers,residual,distinct) or None.
    """
    from scipy.spatial import cKDTree
    if len(corners) < 2 or len(truth) < 2:
        return None
    diag = _diag(corners)
    min_sep = max(MINSEP_FLOOR, MINSEP_FRAC * diag)     # min RANSAC seed corner-pair span
    # A true refine is a small correction of the ~raster-accurate seat; a seed jumping
    # further than this is a coincidental match to a look-alike cluster elsewhere -> reject.
    max_jump = 1.5 * max(SHIFT_FLOOR, SHIFT_FRAC * diag)
    pca_r = max(NEAR_FLOOR, 0.6 * diag)                 # own-parcel neighbourhood for PCA
    tree = cKDTree(truth)
    tdist = np.hypot(*(truth[:, None, :] - truth[None, :, :]).transpose(2, 0, 1))
    best = None
    cc0 = corners.mean(0)

    def _consider(R, t):
        nonlocal best
        if float(np.hypot(*((R @ cc0 + t) - cc0))) > max_jump:
            return
        R2, t2 = _icp(corners, tree, truth, R, t, tol)
        proj = (R2 @ corners.T).T + t2
        dist, idx = tree.query(proj)
        inl_mask = dist <= tol
        inl = int(inl_mask.sum())
        distinct = len(set(idx[inl_mask].tolist()))
        res = float(dist[inl_mask].mean()) if inl else 1e9
        if best is None or (inl, -res) > (best[2], -best[3]):
            best = (R2, t2, inl, res, distinct)

    # PCA global-pose seeds against the stones local to the plot (own-parcel neighbourhood).
    local = truth[np.hypot(*(truth - cc0).T) < pca_r]
    for R, t in _pca_seeds(corners, local):
        _consider(R, t)

    for i, j in combinations(range(len(corners)), 2):
        if float(np.hypot(*(corners[i] - corners[j]))) < min_sep:
            continue
        aa, bb = np.where(np.abs(tdist - float(np.hypot(*(corners[i] - corners[j])))) < tol)
        for a, b in zip(aa, bb):
            if a == b:
                continue
            fit = _rigid_2pt(corners[i], corners[j], truth[a], truth[b])
            if fit is None:
                continue
            _consider(fit[0], fit[1])
    return best


def refine_to_stones(
    results: list[ClubResult],
    truth_stones: np.ndarray,
    *,
    tol: float = TOL,
    min_inliers: int = MIN_INLIERS,
    max_residual: float = MAX_RESIDUAL,
    skip: set | None = None,
) -> RefineStats:
    """Snap each placed FMB onto the surveyor stones when a confident congruent fit exists.

    Mutates ``placement.R/.t/.adjusted`` for refined plots (rigid compose) so all clubbed
    deliverables written afterwards carry the true-stone positions. Returns RefineStats
    (``anchored`` = surveys now on real stones -> treat as fixed downstream).
    """
    truth = np.asarray(truth_stones, float)
    placed = [r for r in results if r.placed and r.placement is not None
              and len(r.placement.corner_points()) >= 3]
    stats = RefineStats(n_plots=len(placed))
    if len(truth) < 2 or not placed:
        return stats

    from shapely.geometry import Polygon

    def _poly(ring):
        p = Polygon([(float(x), float(y)) for x, y in ring])
        if not p.is_valid:
            p = p.buffer(0)
        return p if (not p.is_empty and p.area > 0) else None

    # Phase 1: compute a candidate fit per plot (no mutation yet).
    skip = skip or set()
    cands = []                                    # (r, R_r, t_r, inl, res, refined_ring)
    for r in placed:
        if r.survey_number in skip:
            continue
        pl = r.placement
        corners = pl.corner_points().astype(float)
        c = corners.mean(0)
        diag = _diag(corners)
        near_radius = max(NEAR_FLOOR, NEAR_FRAC * diag)      # size-relative search window
        max_shift = max(SHIFT_FLOOR, SHIFT_FRAC * diag)      # size-relative mis-match bound
        near = truth[np.hypot(*(truth - c).T) < near_radius]
        if len(near) < 2:
            stats.skipped.append((r.survey_number, "no surveyor stones nearby"))
            continue
        fit = _best_fit(corners, near, tol)
        if fit is None:
            stats.skipped.append((r.survey_number, "no rigid fit"))
            continue
        R_r, t_r, inl, res, distinct = fit
        frac = inl / len(corners)
        shift = float(np.hypot(*((R_r @ c + t_r) - c)))
        ok = (inl >= min_inliers and frac >= MIN_INLIER_FRAC and distinct >= min_inliers
              and res <= max_residual and shift <= max_shift)
        if not ok:
            stats.skipped.append(
                (r.survey_number,
                 f"weak fit (inl {inl}/{len(corners)}, res {res:.1f}m, shift {shift:.0f}m)"))
            continue
        cands.append((r, R_r, t_r, inl, res, shift, (R_r @ corners.T).T + t_r))

    # Phase 2: commit greedily, best fit first, but REJECT any snap that would STACK a plot
    # on another (real parcels tile -- share edges, never interiors). This stops a tiny
    # sliver from matching a neighbour's stones and landing inside it.
    footprints = {id(r): (r.placement.footprint() if r.placement else None) for r in placed}
    cands.sort(key=lambda x: (-x[3], x[4]))
    for r, R_r, t_r, inl, res, shift, refined_ring in cands:
        new_fp = _poly(refined_ring)
        conflict = None
        if new_fp is not None:
            for other in placed:
                if other is r:
                    continue
                ofp = footprints.get(id(other))
                if ofp is None or not new_fp.intersects(ofp):
                    continue
                ov = new_fp.intersection(ofp).area / max(min(new_fp.area, ofp.area), 1e-9)
                if ov > OVERLAP_MAX:
                    conflict = (other.survey_number, ov)
                    break
        if conflict is not None:
            stats.skipped.append(
                (r.survey_number, f"snap would overlap {conflict[0]} by {conflict[1]:.0%}"))
            continue
        pl = r.placement
        pl.R = R_r @ np.asarray(pl.R, float)
        pl.t = R_r @ np.asarray(pl.t, float) + t_r
        pl.adjusted = (R_r @ pl.adjusted.T).T + t_r
        footprints[id(r)] = new_fp
        stats.n_refined += 1
        stats.anchored.add(r.survey_number)
        stats.per_plot[r.survey_number] = (inl, res, shift)

    _log.info("stone_refine: %d/%d plots snapped to surveyor stones (anchored); %d kept "
              "cadastre seat", stats.n_refined, stats.n_plots, len(stats.skipped))
    return stats


def resolve_overlaps(results: list[ClubResult], originals: dict, anchored: set,
                     max_overlap: float = OVERLAP_MAX) -> list[tuple[str, str]]:
    """Guarantee a non-overlapping tiling AFTER the whole refine/propagate/snap.

    Any two placed plots overlapping (interior) by more than ``max_overlap`` of the smaller
    footprint are un-tiled: the WORSE plot is reverted to its clean cadastre placement
    (``originals``). "Worse" = unanchored before anchored, then smaller area (a sliver that
    slid onto a real parcel). Reverting to the cadastre seat is safe -- that set was already
    a non-overlapping tiling. Returns the list of (survey, reverted-against) actions.
    """
    from shapely.geometry import Polygon

    def _poly(pl):
        return pl.footprint() if pl is not None else None

    placed = [r for r in results if r.placed and r.placement is not None
              and len(r.placement.corner_points()) >= 3]
    def _score(r):                      # lower = revert first
        return (r.survey_number in anchored,
                _poly(r.placement).area if _poly(r.placement) else 0.0)

    reverts, reverted_once, demoted = [], set(), []
    for _ in range(2 * len(placed) + 2):
        placed = [r for r in results if r.placed and r.placement is not None
                  and len(r.placement.corner_points()) >= 3]
        fps = {id(r): _poly(r.placement) for r in placed}
        worst = None
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                a, b = placed[i], placed[j]
                pa, pb = fps[id(a)], fps[id(b)]
                if pa is None or pb is None or not pa.intersects(pb):
                    continue
                ov = pa.intersection(pb).area / max(min(pa.area, pb.area), 1e-9)
                if ov > max_overlap and (worst is None or ov > worst[0]):
                    worst = (ov, a, b)
        if worst is None:
            break
        _ov, a, b = worst
        victim = a if _score(a) <= _score(b) else b
        keep = b if victim is a else a
        sn = victim.survey_number
        if sn not in reverted_once and originals.get(sn) is not None:
            # First try: drop the plot back to its clean cadastre seat.
            victim.placement.R, victim.placement.t, victim.placement.adjusted = originals[sn]
            anchored.discard(sn)
            reverted_once.add(sn)
            reverts.append((sn, keep.survey_number))
        else:
            # Revert didn't separate them (the neighbour genuinely occupies this space --
            # a cadastre/survey data conflict, e.g. a sliver re-surveyed into its parent):
            # demote to REVIEW so the ACCEPT set stays a clean, honest tiling.
            victim.recommendation = "REVIEW"
            anchored.discard(sn)
            demoted.append((sn, keep.survey_number))
    if reverts or demoted:
        _log.info("resolve_overlaps: %d reverted to cadastre, %d demoted to REVIEW "
                  "(data conflict): reverts=%s demoted=%s",
                  len(reverts), len(demoted), reverts, demoted)
    return reverts + demoted
