"""Relative shared-edge alignment for the new M2 (FMB-only georeference + club).

WHY THIS EXISTS (vs. boundary_snap)
-----------------------------------
``boundary_snap`` snaps individual coincident CORNERS within a few metres. But when a
plot is seated on its own (approximate, raster) cadastral parcel, it can land a whole
10-30 m offset from a neighbour that it truly shares a boundary with -- every shared
corner is off by the SAME vector. That is a rigid TRANSLATION error, which corner-snap
cannot fix (its centroid guard reverts any uniform shift). This pass is the automated
"FMBS_STONES_MATCH": it detects a genuine shared boundary (two near-parallel edges that
overlap in projection -- corroborated because BOTH FMBs independently drew that edge)
and translates the plots toward a common line so their boundaries MERGE.

0-FALSE-POSITIVE / RIGID DISCIPLINE
-----------------------------------
* TRANSLATION ONLY -- shape, scale and rotation are untouched, so geometry stays exactly
  as placed (the FMB is never warped).
* A pair is a shared-edge constraint ONLY if the closest edges are near-parallel
  (``ang_tol``) AND overlap in projection by ``ovl_min`` of the shorter edge AND the gap
  is below ``d_max``. This excludes corner-only / perpendicular adjacencies (a plot that
  merely touches a corner of a big neighbour is NOT dragged onto it).
* Each plot's cumulative move is capped at ``move_cap``; a corroborated shared edge only
  refines the ~raster-accurate absolute seat, it never relocates a parcel.
* After solving, any move that creates a NEW interior overlap above ``new_overlap_max``
  is reverted for the plot that moved further -- real parcels tile, they do not stack.

Only ACCEPT / ACCEPT_SEEDED plots move; REVIEW / NO_COVERAGE are never touched.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .placement import ClubResult

_log = logging.getLogger(__name__)

# PRINCIPLED, SIZE-INDEPENDENT gates (pure geometry -- identical on any village):
ANG_TOL = 10.0          # deg: edges must be this parallel to be one shared boundary
OVL_MIN = 0.50          # shared edges must overlap >= this fraction of the shorter edge
NEW_OVERLAP_MAX = 0.05  # a move creating more new overlap than this (of smaller) is undone
STEP = 0.5             # relaxation step (fraction of the residual applied per sweep)
ITERS = 40             # relaxation sweeps
# SIZE-RELATIVE distances -- expressed as a FRACTION of the plot's own size (diagonal), with an
# absolute floor for the smallest plots, so nothing is tuned to one village's metre-scale. A gap
# is a "shared boundary" only if it is small RELATIVE to the plots it joins; a plot may only be
# nudged a small fraction of its own size (a bigger move is relocation, not edge-sharing).
D_MAX_FRAC = 0.25       # max shared-edge gap as a fraction of the smaller plot's diagonal
D_MAX_FLOOR = 12.0      # m: floor so tiny plots still admit a real shared edge
MOVE_FRAC = 0.15        # max translation as a fraction of the plot's own diagonal
MOVE_FLOOR = 12.0       # m: floor for small plots
# Back-compat scalar defaults (used only if a caller passes an explicit absolute cap).
MOVE_CAP = 25.0


def _diag(ring: np.ndarray) -> float:
    """Plot size = bounding-box diagonal of its corner ring (village-scale-independent)."""
    if len(ring) < 2:
        return 0.0
    return float(np.hypot(np.ptp(ring[:, 0]), np.ptp(ring[:, 1])))


@dataclass
class AlignStats:
    n_plots: int = 0
    n_constraints: int = 0
    n_moved: int = 0
    max_move: float = 0.0
    reverted: list[tuple[str, str]] = field(default_factory=list)


def _ring_pts(pl) -> np.ndarray:
    """Absolute UTM corner-ring points of a placement (>=3) or empty."""
    pts = pl.corner_points()
    return pts if len(pts) >= 3 else np.empty((0, 2))


def _edges(ring: np.ndarray):
    n = len(ring)
    return [(ring[i], ring[(i + 1) % n]) for i in range(n)]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-9 else v


def _ang180(a: np.ndarray, b: np.ndarray) -> float:
    """Undirected angle (deg, 0..90) between two segment directions."""
    da = np.degrees(np.arctan2(a[1], a[0])) % 180.0
    db = np.degrees(np.arctan2(b[1], b[0])) % 180.0
    d = abs(da - db) % 180.0
    return min(d, 180.0 - d)


def _proj_overlap(a1, a2, b1, b2) -> float:
    """Fraction of the shorter edge that the two segments overlap along a's direction."""
    d = _unit(a2 - a1)
    ta1, ta2 = 0.0, float(np.dot(a2 - a1, d))
    tb1, tb2 = float(np.dot(b1 - a1, d)), float(np.dot(b2 - a1, d))
    lo_a, hi_a = min(ta1, ta2), max(ta1, ta2)
    lo_b, hi_b = min(tb1, tb2), max(tb1, tb2)
    ov = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    shorter = max(min(hi_a - lo_a, hi_b - lo_b), 1e-9)
    return ov / shorter


def _shared_edge(ring_i: np.ndarray, ring_j: np.ndarray, d_max: float):
    """Best shared-boundary edge pair between two rings, or None.

    ``d_max`` (size-relative, set by the caller from the plots' own diagonals) is the
    largest gap that still counts as a shared boundary.

    Returns (normal_unit, signed_gap, weight) where ``normal_unit`` is the unit
    perpendicular to plot i's edge and ``signed_gap`` is the distance to move plot i
    ALONG that normal to reach plot j's edge line (so i moves +g/2, j moves -g/2).
    ``weight`` is the overlap length (longer shared edges pull harder).
    """
    from shapely.geometry import LineString
    best = None
    for a1, a2 in _edges(ring_i):
        for b1, b2 in _edges(ring_j):
            if _ang180(a2 - a1, b2 - b1) > ANG_TOL:
                continue
            if _proj_overlap(a1, a2, b1, b2) < OVL_MIN:
                continue
            gap = LineString([a1, a2]).distance(LineString([b1, b2]))
            if gap > d_max or gap < 1e-6:
                continue
            if best is None or gap < best[0]:
                # unit normal to edge i; sign it so it points toward edge j.
                d = _unit(a2 - a1)
                n = np.array([-d[1], d[0]])
                mid_i = 0.5 * (a1 + a2)
                mid_j = 0.5 * (b1 + b2)
                signed = float(np.dot(mid_j - mid_i, n))
                if signed < 0:
                    n, signed = -n, -signed
                ov_len = _proj_overlap(a1, a2, b1, b2) * min(
                    np.hypot(*(a2 - a1)), np.hypot(*(b2 - b1)))
                best = (gap, n, signed, ov_len)
    if best is None:
        return None
    _gap, n, signed, w = best
    return n, signed, w


def align_shared_edges(
    results: list[ClubResult],
    *,
    move_cap: float | None = None,
    new_overlap_max: float = NEW_OVERLAP_MAX,
    fixed: set[str] | None = None,
    recs: tuple[str, ...] | None = None,
) -> AlignStats:
    """Translate ACCEPT plots toward their corroborated shared boundaries (rigid).

    Mutates ``placement.t`` and ``placement.adjusted`` (both by the same vector, so the
    whole plot -- corners, interior stones and every entity written from ``t`` -- moves
    together). Returns :class:`AlignStats`. 0-FP: translation-only, gated by parallel +
    overlap corroboration, capped, and overlap-reverted. Every distance threshold is
    RELATIVE to each plot's own size (see the module constants), so the pass behaves
    identically on a village of 40 m urban plots or 300 m rural ones -- nothing is tuned
    to one dataset. ``move_cap`` (optional) only imposes a further absolute ceiling.

    ``fixed`` is a set of survey numbers that are ANCHORED (already snapped to the real
    surveyor stones): they never move, and a neighbour sharing an edge with an anchor
    takes the FULL correction toward it -- so accurate placement PROPAGATES outward from
    the stone-matched plots (the client's relative FMBS_STONES_MATCH).

    ``recs`` widens which recommendations participate (default: the placed ACCEPT set).
    Club-all M2 passes ``("ACCEPT", "ACCEPT_SEEDED", "REVIEW")`` with ``fixed`` = the
    confident set, so a LOW-confidence plot takes the full gap toward its confident
    neighbours (the corroborated reseat) while the confident tiling itself never moves.
    All the same guards apply: translation-only, size-relative caps, overlap-revert.
    """
    if recs is None:
        placed = [r for r in results if r.placed and r.placement is not None
                  and len(_ring_pts(r.placement)) >= 3]
    else:
        placed = [r for r in results if r.recommendation in recs
                  and r.placement is not None and len(_ring_pts(r.placement)) >= 3]
    stats = AlignStats(n_plots=len(placed))
    if len(placed) < 2:
        return stats
    fixed_idx = {k for k in range(len(placed))
                 if placed[k].survey_number in (fixed or set())}

    rings0 = [_ring_pts(r.placement).astype(float) for r in placed]
    cents = [ring.mean(axis=0) for ring in rings0]
    diags = [_diag(ring) for ring in rings0]
    # Per-plot cap = a fraction of the plot's OWN size (floored); an explicit ``move_cap``
    # only lowers it further. Nothing here is a fixed metre value tuned to one village.
    caps = [max(MOVE_FLOOR, MOVE_FRAC * d) for d in diags]
    if move_cap is not None:
        caps = [min(move_cap, c) for c in caps]

    # --- 1. Fix the shared-edge pairing ONCE from the initial placement.
    #        constraints: (i, j, n_i, signed_gap0, weight)  where n_i is i's edge normal.
    constraints = []
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            # Two plots can only share an edge if their centroids are within roughly the
            # sum of their sizes -- a scale-free neighbour prune (was a fixed 550 m).
            if np.hypot(*(cents[i] - cents[j])) > (diags[i] + diags[j]):
                continue
            d_max = max(D_MAX_FLOOR, D_MAX_FRAC * min(diags[i], diags[j]))
            se = _shared_edge(rings0[i], rings0[j], d_max)
            if se is None:
                continue
            n_i, signed, w = se
            constraints.append((i, j, n_i, signed, w))
    stats.n_constraints = len(constraints)
    if not constraints:
        return stats

    # --- 2. Gauss-Seidel relaxation for a per-plot translation. Each sweep, every plot
    #         moves a STEP fraction of the weighted-average residual gap to its shared
    #         edges (half the current gap, since the neighbour takes the other half).
    t = [np.zeros(2) for _ in placed]
    for _ in range(ITERS):
        max_delta = 0.0
        acc = [np.zeros(2) for _ in placed]
        wsum = [0.0 for _ in placed]
        for (i, j, n_i, signed0, w) in constraints:
            # current perpendicular gap along n_i after applying t_i, t_j.
            cur_gap = signed0 - float(np.dot(t[i] - t[j], n_i))
            i_fixed, j_fixed = i in fixed_idx, j in fixed_idx
            if i_fixed and j_fixed:
                continue
            if i_fixed:                       # j moves the FULL gap toward the anchor i
                acc[j] += w * (-cur_gap * n_i)
                wsum[j] += w
            elif j_fixed:                     # i moves the FULL gap toward the anchor j
                acc[i] += w * (cur_gap * n_i)
                wsum[i] += w
            else:                             # both free -> split the gap
                acc[i] += w * (0.5 * cur_gap * n_i)
                wsum[i] += w
                acc[j] += w * (-0.5 * cur_gap * n_i)
                wsum[j] += w
        for k in range(len(placed)):
            if k in fixed_idx or wsum[k] <= 0:
                continue
            delta = STEP * (acc[k] / wsum[k])
            t[k] = t[k] + delta
            # cap cumulative magnitude at this plot's OWN size-relative cap
            mag = float(np.hypot(*t[k]))
            if mag > caps[k]:
                t[k] = t[k] * (caps[k] / mag)
            max_delta = max(max_delta, float(np.hypot(*delta)))
        if max_delta < 0.05:
            break

    # --- 3. Tentatively apply, then revert any move that creates a NEW overlap.
    from shapely.geometry import Polygon

    def _poly(ring):
        p = Polygon([(float(x), float(y)) for x, y in ring])
        if not p.is_valid:
            p = p.buffer(0)
        return p if (not p.is_empty and p.area > 0) else None

    pre = [_poly(r) for r in rings0]
    post = [_poly(rings0[k] + t[k]) for k in range(len(placed))]

    moved = {k for k in range(len(placed)) if float(np.hypot(*t[k])) > 1e-3}
    changed = True
    guard = 0
    while changed and guard < len(placed) + 2:
        changed = False
        guard += 1
        for a in range(len(placed)):
            for b in range(a + 1, len(placed)):
                if a not in moved and b not in moved:
                    continue
                pa, pb = post[a], post[b]
                if pa is None or pb is None or not pa.intersects(pb):
                    continue
                ov = pa.intersection(pb).area / max(min(pa.area, pb.area), 1e-9)
                ra, rb = pre[a], pre[b]
                ov0 = 0.0
                if ra is not None and rb is not None and ra.intersects(rb):
                    ov0 = ra.intersection(rb).area / max(min(ra.area, rb.area), 1e-9)
                if ov - ov0 <= new_overlap_max:
                    continue
                cand = [k for k in (a, b) if k in moved]
                victim = max(cand, key=lambda k: float(np.hypot(*t[k])))
                stats.reverted.append((placed[victim].survey_number,
                                       f"new overlap {ov - ov0:.0%} with "
                                       f"{placed[a if victim == b else b].survey_number}"))
                t[victim] = np.zeros(2)
                post[victim] = pre[victim]
                moved.discard(victim)
                changed = True

    # --- 4. Commit surviving translations onto the placements.
    for k in range(len(placed)):
        tk = t[k]
        if float(np.hypot(*tk)) <= 1e-3:
            continue
        pl = placed[k].placement
        pl.t = np.asarray(pl.t, float) + tk
        pl.adjusted = pl.adjusted + tk
        stats.n_moved += 1
        stats.max_move = max(stats.max_move, float(np.hypot(*tk)))

    _log.info("edge_align: %d plots, %d shared-edge constraints, %d plots moved "
              "(max %.1f m), %d reverted", stats.n_plots, stats.n_constraints,
              stats.n_moved, stats.max_move, len(stats.reverted))
    return stats
