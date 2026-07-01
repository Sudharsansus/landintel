"""Inter-plot boundary snapping for the new M2 (FMB-only georeference + club).

WHY THIS EXISTS
---------------
Each ACCEPT FMB is seated RIGIDLY on its OWN cadastral parcel (or GPS / propagated
neighbour) independently. Two adjacent plots therefore land *near* a common boundary
but almost never on the EXACT same edge -- their shared corners differ by a metre or
so, leaving hairline slivers / gaps instead of one coincident edge. The manual workers'
"FMBS_STONES_MATCH" step snaps a new plot's corner stones onto the known stone positions
of its neighbours so adjacent plots share ONE physical boundary. This module is the
automated equivalent, applied as a QUALITY pass AFTER placement -- never a re-placement.

WHAT IT DOES
------------
1. Collect the absolute-UTM corner positions of every placed (ACCEPT/ACCEPT_SEEDED)
   plot's outer ring.
2. CLUSTER corners that lie within ``tol`` metres of each other ACROSS plots
   (union-find), so a stone shared by 2-3 neighbours forms one cluster.
3. SNAP every corner in a multi-plot cluster to the cluster centroid -- giving the
   neighbours an EXACT coincident corner (and hence a coincident shared edge).

0-FALSE-POSITIVE / RIGID-GEOMETRY DISCIPLINE
--------------------------------------------
This pass must never re-place a plot or warp it. It only nudges shared corners, and
only a little. Each plot is snapped, then VALIDATED; any plot that violates a guard is
fully REVERTED (its original placement is restored), so the worst case is "no change":

  (a) per-corner displacement is capped at ``tol`` -- a cluster member further than
      ``tol`` from the centroid is dropped from that cluster (not dragged across).
  (b) after snapping, the plot ring must stay CLOSED and SIMPLE (no self-intersection),
      and its area must not change by more than ``max_area_frac``.
  (c) the plot centroid must not move more than ``max_centroid_move`` metres (the
      validated cadastral seat must be preserved -- snapping shares edges, it does not
      relocate parcels).
  (d) the snap must not create a NEW interior footprint overlap above
      ``new_overlap_max`` with any other placed plot.

Only the corner positions inside ``CandidatePlacement.adjusted`` are touched (the ring
indices); R/s/t and every interior stone are left exactly as placed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .placement import ClubResult

_log = logging.getLogger(__name__)

# Default corner-cluster radius (m). Two corners of different plots within this are the
# same physical stone. Kept >= relative_club.EDGE_COINCIDE_TOL so anything the club
# already treats as a shared edge is snap-eligible.
DEFAULT_TOL = 5.0
# A snapped plot whose centroid wanders more than this from its placed centroid is a
# re-placement, not an edge snap -> revert it.
MAX_CENTROID_MOVE = 3.0
# Allowed fractional change in a plot's footprint area after snapping (rigid-ish guard).
MAX_AREA_FRAC = 0.05
# A snap that creates an interior overlap above this fraction of the smaller footprint
# is making plots overlap, not tile -> revert the snap.
NEW_OVERLAP_MAX = 0.05
# A corner whose target differs from its current position by more than this (m) counts
# as actually moved. ABSOLUTE -- never use np.allclose's relative tol on UTM coords
# (~7e5), where rtol=1e-5 would call a ~7 m move "close" and silently skip every snap.
_MOVED_EPS = 1e-4


def _moved(a: np.ndarray, b: np.ndarray) -> bool:
    """True if point a is more than _MOVED_EPS metres from b (absolute, UTM-safe)."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1])) > _MOVED_EPS


@dataclass
class SnapStats:
    """What the quality pass did, for reporting / asserts."""
    n_plots: int = 0
    n_clusters: int = 0
    n_corners_snapped: int = 0
    max_corner_move: float = 0.0
    max_centroid_move: float = 0.0
    skipped: list[tuple[str, str]] = None   # [(survey, reason)]
    # Relative shared-edge alignment (edge_align, runs before the corner snap).
    n_edge_constraints: int = 0
    n_edge_moved: int = 0
    max_edge_move: float = 0.0
    # Plots snapped to the real surveyor stones (stone_refine, when ground truth supplied).
    n_anchored: int = 0

    def __post_init__(self):
        if self.skipped is None:
            self.skipped = []


# ---------------------------------------------------------------------------
# Union-find over corner endpoints.
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _ring_indices(placement) -> list[int]:
    """Valid ring corner indices into placement.adjusted (deduped, in ring order)."""
    adj = placement.adjusted
    seen: set[int] = set()
    out: list[int] = []
    for i in placement.corner_ring:
        if 0 <= i < len(adj) and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _polygon(pts: np.ndarray):
    from shapely.geometry import Polygon
    if len(pts) < 3:
        return None
    poly = Polygon([(float(x), float(y)) for x, y in pts])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def _footprints(results: list[ClubResult]) -> dict[int, object]:
    return {id(r): (r.placement.footprint() if r.placement is not None else None)
            for r in results}


# ---------------------------------------------------------------------------
# The snap.
# ---------------------------------------------------------------------------

def snap_shared_boundaries(
    results: list[ClubResult],
    tol: float = DEFAULT_TOL,
    *,
    max_centroid_move: float = MAX_CENTROID_MOVE,
    max_area_frac: float = MAX_AREA_FRAC,
    new_overlap_max: float = NEW_OVERLAP_MAX,
    fixed: set[str] | None = None,
) -> SnapStats:
    """Snap coincident corners of adjacent PLACED plots onto one common position.

    Mutates ``r.placement.adjusted`` in place for placed (ACCEPT/ACCEPT_SEEDED) plots so
    that adjacent plots share EXACT coincident corners -> a common boundary edge, no
    slivers. REVIEW / NO_COVERAGE plots are never touched.

    Returns a :class:`SnapStats`. Any plot that would violate a guard (corner move > tol,
    broken/over-warped ring, centroid drift, or a new footprint overlap) is fully reverted
    to its pre-snap placement, so the pass can only improve or no-op a plot -- never
    corrupt it (0-FP preserved).
    """
    placed = [r for r in results
              if r.placed and r.placement is not None and len(_ring_indices(r.placement)) >= 3]
    stats = SnapStats(n_plots=len(placed))
    if len(placed) < 2:
        return stats
    fixed_pi = {pi for pi in range(len(placed))
                if placed[pi].survey_number in (fixed or set())}

    # --- 1. Flatten every placed plot's ring corners into one point list, remembering
    #         which (plot, adjusted-index) each came from.
    pts: list[tuple[float, float]] = []
    owner: list[tuple[int, int]] = []      # (plot_pos_in_`placed`, adjusted_index)
    for pi, r in enumerate(placed):
        adj = r.placement.adjusted
        for ai in _ring_indices(r.placement):
            pts.append((float(adj[ai][0]), float(adj[ai][1])))
            owner.append((pi, ai))
    P = np.asarray(pts, float)
    n = len(P)

    # --- 2. Union corners within `tol` (across plots). Use a coarse grid bucket so we
    #         only test nearby pairs (O(n) on real village-scale clouds).
    uf = _UF(n)
    cell = max(tol, 1e-6)
    buckets: dict[tuple[int, int], list[int]] = {}
    for k in range(n):
        gx, gy = int(P[k, 0] // cell), int(P[k, 1] // cell)
        buckets.setdefault((gx, gy), []).append(k)
    tol2 = tol * tol
    for k in range(n):
        gx, gy = int(P[k, 0] // cell), int(P[k, 1] // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((gx + dx, gy + dy), ()):
                    if j <= k:
                        continue
                    if owner[j][0] == owner[k][0]:
                        continue            # never merge two corners of the SAME plot
                    d2 = (P[k, 0] - P[j, 0]) ** 2 + (P[k, 1] - P[j, 1]) ** 2
                    if d2 <= tol2:
                        uf.union(k, j)

    # --- 3. Build clusters; keep only those spanning >= 2 distinct plots.
    clusters: dict[int, list[int]] = {}
    for k in range(n):
        clusters.setdefault(uf.find(k), []).append(k)

    # Target position per corner index k (default: stay put).
    target = P.copy()
    cluster_count = 0
    for members in clusters.values():
        plots_in = {owner[k][0] for k in members}
        if len(plots_in) < 2:
            continue
        c = P[members].mean(axis=0)
        # Guard (a): drop members further than tol from the centroid (don't drag across).
        kept = [k for k in members
                if (P[k, 0] - c[0]) ** 2 + (P[k, 1] - c[1]) ** 2 <= tol2]
        if len({owner[k][0] for k in kept}) < 2:
            continue
        c = P[kept].mean(axis=0)           # recompute centroid on the kept subset
        # Re-guard after recompute (centroid can shift): only keep within tol of it.
        kept = [k for k in kept
                if (P[k, 0] - c[0]) ** 2 + (P[k, 1] - c[1]) ** 2 <= tol2]
        if len({owner[k][0] for k in kept}) < 2:
            continue
        c = P[kept].mean(axis=0)
        cluster_count += 1
        # If any corner in the cluster belongs to an ANCHORED (stone-matched) plot, the
        # shared point is the anchor's position -- non-anchored neighbours snap ONTO it,
        # anchored corners stay put (never dragged off the surveyor stones).
        fixed_ks = [k for k in kept if owner[k][0] in fixed_pi]
        anchor = P[fixed_ks].mean(axis=0) if fixed_ks else c
        for k in kept:
            if owner[k][0] in fixed_pi:
                continue
            target[k] = anchor
    stats.n_clusters = cluster_count

    if cluster_count == 0:
        return stats

    # --- 4. Apply per plot, validating each; revert any plot that fails a guard.
    #         Stash original adjusted arrays so a revert is exact.
    originals = {pi: placed[pi].placement.adjusted.copy() for pi in range(len(placed))}
    pre_centroids = {pi: np.asarray(placed[pi].placement.centroid(), float)
                     for pi in range(len(placed))}
    pre_areas = {}
    for pi, r in enumerate(placed):
        fp = r.placement.footprint()
        pre_areas[pi] = fp.area if fp is not None else 0.0

    # Group target moves by plot (anchored plots never move).
    moves_by_plot: dict[int, list[tuple[int, np.ndarray]]] = {}
    for k in range(n):
        pi, ai = owner[k]
        if pi in fixed_pi:
            continue
        if _moved(target[k], P[k]):
            moves_by_plot.setdefault(pi, []).append((ai, target[k]))

    snapped_plots: list[int] = []
    for pi, moves in moves_by_plot.items():
        r = placed[pi]
        adj = r.placement.adjusted          # mutate in place
        # Apply moves.
        worst = 0.0
        for ai, tgt in moves:
            worst = max(worst, float(np.hypot(adj[ai][0] - tgt[0], adj[ai][1] - tgt[1])))
            adj[ai] = tgt

        ok, reason = _validate_plot(
            r, originals[pi], pre_centroids[pi], pre_areas[pi],
            tol, max_centroid_move, max_area_frac)
        if not ok:
            r.placement.adjusted = originals[pi].copy()   # full revert
            stats.skipped.append((r.survey_number, reason))
            continue
        snapped_plots.append(pi)
        stats.max_corner_move = max(stats.max_corner_move, worst)

    # --- 5. Global overlap guard: a snap that pushed two plots into each other beyond
    #         `new_overlap_max` (and was not already overlapping) is undone for the plot
    #         whose centroid moved more (keep the better-anchored one put).
    _undo_new_overlaps(results, placed, originals, pre_centroids,
                       snapped_plots, new_overlap_max, stats)

    # --- 6. Recompute reported centroid moves on the FINAL placements.
    for pi in range(len(placed)):
        cur = np.asarray(placed[pi].placement.centroid(), float)
        stats.max_centroid_move = max(
            stats.max_centroid_move, float(np.hypot(*(cur - pre_centroids[pi]))))
    snapped_set = set(snapped_plots)
    stats.n_corners_snapped = sum(
        1 for k in range(n)
        if owner[k][0] in snapped_set and _moved(target[k], P[k]))

    _log.info("boundary_snap: %d plots, %d shared-corner clusters, %d corners snapped; "
              "max corner move %.2f m, max centroid move %.2f m, %d plots skipped",
              stats.n_plots, stats.n_clusters, stats.n_corners_snapped,
              stats.max_corner_move, stats.max_centroid_move, len(stats.skipped))
    return stats


def _validate_plot(r, original, pre_centroid, pre_area,
                   tol, max_centroid_move, max_area_frac) -> tuple[bool, str]:
    """Check a freshly-snapped plot against the rigid-preservation guards."""
    pl = r.placement
    ring = _ring_indices(pl)
    pts = pl.adjusted[np.array(ring)]

    # Per-corner displacement cap (belt-and-braces; clustering already capped at tol).
    orig_ring = original[np.array(ring)]
    moves = np.hypot(pts[:, 0] - orig_ring[:, 0], pts[:, 1] - orig_ring[:, 1])
    if float(moves.max(initial=0.0)) > tol + 1e-6:
        return False, f"corner move {float(moves.max()):.2f} m > tol {tol:.1f}"

    # Ring must stay closed + simple (shapely .is_valid covers self-intersection).
    poly = _polygon(pts)
    if poly is None:
        return False, "ring degenerate after snap"
    from shapely.geometry import Polygon
    raw = Polygon([(float(x), float(y)) for x, y in pts])
    if not raw.is_valid or not raw.is_simple:
        return False, "ring self-intersects after snap"

    # Area must not change much (rigid-ish).
    if pre_area > 0:
        frac = abs(poly.area - pre_area) / pre_area
        if frac > max_area_frac:
            return False, f"area changed {frac:.0%} (> {max_area_frac:.0%})"

    # Centroid must not relocate the parcel.
    cur_c = np.asarray(pl.centroid(), float)
    cmove = float(np.hypot(*(cur_c - pre_centroid)))
    if cmove > max_centroid_move:
        return False, f"centroid moved {cmove:.2f} m (> {max_centroid_move:.1f})"
    return True, ""


def _undo_new_overlaps(results, placed, originals, pre_centroids,
                       snapped_plots, new_overlap_max, stats) -> None:
    """If snapping created a NEW interior overlap above threshold between two placed
    plots, revert the snap on the plot whose centroid moved more (lesser-anchored)."""
    from shapely.geometry import Polygon  # noqa: F401  (footprint uses shapely)

    snapped = set(snapped_plots)
    if not snapped:
        return

    # Pre-snap footprints (from originals) and post-snap footprints (current).
    def _fp(pts):
        return _polygon(pts)

    def _centroid_drift(pi):
        cur = np.asarray(placed[pi].placement.centroid(), float)
        return float(np.hypot(*(cur - pre_centroids[pi])))

    pre_fp, post_fp = {}, {}
    for pi, r in enumerate(placed):
        ring = _ring_indices(r.placement)
        idx = np.array(ring)
        pre_fp[pi] = _fp(originals[pi][idx])
        post_fp[pi] = _fp(r.placement.adjusted[idx])

    changed = True
    guard = 0
    while changed and guard < len(placed) + 1:
        changed = False
        guard += 1
        for a in range(len(placed)):
            for b in range(a + 1, len(placed)):
                if a not in snapped and b not in snapped:
                    continue
                pa, pb = post_fp[a], post_fp[b]
                if pa is None or pb is None or not pa.intersects(pb):
                    continue
                ov = pa.intersection(pb).area / max(min(pa.area, pb.area), 1e-9)
                # Was this overlap already present before the snap? Then it is not NEW.
                ra, rb = pre_fp[a], pre_fp[b]
                ov0 = 0.0
                if ra is not None and rb is not None and ra.intersects(rb):
                    ov0 = ra.intersection(rb).area / max(min(ra.area, rb.area), 1e-9)
                if ov - ov0 <= new_overlap_max:
                    continue
                # NEW overlap -> revert whichever snapped plot moved its centroid more.
                cand = [pi for pi in (a, b) if pi in snapped]
                if not cand:
                    continue
                victim = max(cand, key=_centroid_drift)
                placed[victim].placement.adjusted = originals[victim].copy()
                ring = _ring_indices(placed[victim].placement)
                post_fp[victim] = _fp(placed[victim].placement.adjusted[np.array(ring)])
                snapped.discard(victim)
                stats.skipped.append(
                    (placed[victim].survey_number,
                     f"snap created new overlap {ov - ov0:.0%} with "
                     f"{placed[a if victim == b else b].survey_number}; reverted"))
                changed = True
    # Keep snapped_plots consistent for the caller's reporting.
    snapped_plots[:] = sorted(snapped)
