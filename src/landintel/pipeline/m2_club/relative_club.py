"""M2 method 3 -- relative clubbing: tie FMBs to each other by their shared edges.

This is the "FMBS_STONES_MATCH" idea (the client's Stage 3), realized WITHOUT a
surveyor file. Two purposes, both 0-FP:

1. CORROBORATION (label-free, geometric). After two FMBs are independently seated
   in absolute UTM (by cadastral or GPS), if a corner-edge of one COINCIDES with a
   corner-edge of the other in absolute space, the two unrelated placements agree
   on a shared physical boundary -> each confirms the other. This needs no OCR
   labels, so wrong labels can only fail to corroborate, never create a false one.

2. PROPAGATION (label-guided, strictly gated). An FMB with NO parcel and NO GPS but
   a neighbour-label naming a SEATED plot can be placed by aligning its shared edge
   onto that neighbour's absolute shared edge. The neighbour-label tells us WHICH
   edge; the alignment is rigid (scale ~1); the orientation that puts the two
   interiors on OPPOSITE sides (real parcels tile) is chosen, and the result must
   not overlap any placed plot. A wrong label fails the non-overlap / scale gate.

Corner-stone POSITIONS are load-bearing; corner LABELS are not (matched on geometry).
"""
from __future__ import annotations

import logging
import math
import re

import numpy as np

from ..m2_georef.extract_m1 import M1PlotData
from ..m2_georef.transform import umeyama
from .placement import CandidatePlacement

_log = logging.getLogger(__name__)

# Two corner endpoints within this distance (m) are the SAME physical stone.
EDGE_COINCIDE_TOL = 3.0
# Shared-edge lengths (A's vs B's measure of the same boundary) must agree within
# this, as max(absolute m, fraction of length).
EDGE_LEN_ABS_TOL = 2.0
EDGE_LEN_REL_TOL = 0.08
# Propagated placement scale must stay ~1 (M1 metres -> UTM metres).
PROP_SCALE_LO = 0.85
PROP_SCALE_HI = 1.18
# Interior overlap (fraction of smaller footprint) above this = the two plots are
# NOT tiling -> reject the propagated orientation.
PROP_MAX_OVERLAP = 0.12
# After the 2-point shared-edge alignment, any further B stone landing within this
# distance (m) of an A stone is the SAME physical corner -> extra correspondence
# for the >=3-point least-squares re-fit (client directive 2026-07-02: match a
# minimum of 3 stone points per FMB wherever the data allows).
PROP_REFIT_SNAP_M = 2.0

_DIGITS = re.compile(r"(\d{1,5})")


def _norm_survey(raw: str | None) -> str | None:
    if raw is None:
        return None
    m = _DIGITS.search(str(raw))
    return m.group(1) if m else None


def _ring_edges(m1: M1PlotData) -> list[tuple[int, int]]:
    """Consecutive corner-stone index pairs around the outer ring."""
    ring = m1.outer_stone_indices
    if len(ring) < 3:
        return []
    return [(ring[k], ring[(k + 1) % len(ring)]) for k in range(len(ring))]


def _rel_xy(m1: M1PlotData, i: int) -> tuple[float, float]:
    return (m1.stones[i].x, m1.stones[i].y)


def _seg_len(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _len_agrees(la: float, lb: float) -> bool:
    return abs(la - lb) <= max(EDGE_LEN_ABS_TOL, EDGE_LEN_REL_TOL * max(la, lb))


def _segments_coincide(p1, p2, q1, q2, tol: float = EDGE_COINCIDE_TOL) -> bool:
    """True if segment (p1,p2) and (q1,q2) are the same edge (either orientation)."""
    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    return ((d(p1, q1) <= tol and d(p2, q2) <= tol)
            or (d(p1, q2) <= tol and d(p2, q1) <= tol))


# ---------------------------------------------------------------------------
# 1. Corroboration: label-free agreement of two already-seated plots.
# ---------------------------------------------------------------------------

def abs_edges(placement: CandidatePlacement) -> list[tuple[tuple, tuple]]:
    """Absolute-UTM corner-edge segments of a placed plot."""
    ring = placement.corner_ring
    adj = placement.adjusted
    segs = []
    if len(ring) < 3:
        return segs
    for k in range(len(ring)):
        a = ring[k]
        b = ring[(k + 1) % len(ring)]
        if 0 <= a < len(adj) and 0 <= b < len(adj):
            segs.append(((float(adj[a][0]), float(adj[a][1])),
                         (float(adj[b][0]), float(adj[b][1]))))
    return segs


def shares_edge(pa: CandidatePlacement, pb: CandidatePlacement) -> bool:
    """True if any absolute corner-edge of pa coincides with one of pb."""
    ea, eb = abs_edges(pa), abs_edges(pb)
    return any(_segments_coincide(p1, p2, q1, q2)
               for (p1, p2) in ea for (q1, q2) in eb)


def corroborate_seated(
    placements: dict[str, CandidatePlacement],
    m1s: dict[str, M1PlotData],
) -> dict[str, list[str]]:
    """For every pair of seated FMBs, record mutual corroboration when a corner-edge
    of one coincides (in absolute UTM) with a corner-edge of the other.

    Returns {survey -> [neighbouring surveys that share a coincident edge]}. Purely
    geometric -- independent of OCR labels -- so it can only confirm correct
    placements, never invent one.
    """
    surveys = [s for s in placements if s in m1s]
    abs_edges: dict[str, list[tuple[tuple, tuple]]] = {}
    for s in surveys:
        pl = placements[s]
        m1 = m1s[s]
        ring = pl.corner_ring
        if len(ring) < 3:
            abs_edges[s] = []
            continue
        adj = pl.adjusted
        segs = []
        for k in range(len(ring)):
            a = ring[k]
            b = ring[(k + 1) % len(ring)]
            if 0 <= a < len(adj) and 0 <= b < len(adj):
                segs.append(((float(adj[a][0]), float(adj[a][1])),
                             (float(adj[b][0]), float(adj[b][1]))))
        abs_edges[s] = segs

    out: dict[str, list[str]] = {s: [] for s in surveys}
    for i in range(len(surveys)):
        for j in range(i + 1, len(surveys)):
            sa, sb = surveys[i], surveys[j]
            shared = False
            for (p1, p2) in abs_edges[sa]:
                for (q1, q2) in abs_edges[sb]:
                    if _segments_coincide(p1, p2, q1, q2):
                        shared = True
                        break
                if shared:
                    break
            if shared:
                out[sa].append(sb)
                out[sb].append(sa)
    return out


# ---------------------------------------------------------------------------
# 2. Propagation: place an un-seated FMB from a seated neighbour by shared edge.
# ---------------------------------------------------------------------------

def _point_seg_dist(px: float, py: float,
                    ax: float, ay: float, bx: float, by: float) -> float:
    """Minimum distance from point (px,py) to the segment (ax,ay)-(bx,by).

    Used to match a neighbour label to the ring edge it lies on. A neighbour label
    can sit anywhere ALONG the shared edge (often near a corner), so matching by the
    edge MIDPOINT mis-picks a perpendicular edge whose midpoint happens to be nearer
    the corner-positioned label; point-to-segment distance is the correct measure and
    is always at least as accurate as midpoint distance."""
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _neighbor_edge(m1: M1PlotData, neighbor_survey: str) -> tuple[int, int, float] | None:
    """The ring edge of ``m1`` that borders ``neighbor_survey`` (from neighbour
    labels), as (corner_a, corner_b, length). None if no label names that survey.
    """
    nbr = _norm_survey(neighbor_survey)
    if nbr is None:
        return None
    targets = [(float(t["x"]), float(t["y"]))
               for t in m1.neighbor_label_texts
               if _norm_survey(t.get("text")) == nbr]
    if not targets:
        return None
    edges = _ring_edges(m1)
    if not edges:
        return None
    best = None  # (dist, a, b)
    for a, b in edges:
        pa, pb = _rel_xy(m1, a), _rel_xy(m1, b)
        for tx, ty in targets:
            # point-to-segment (not midpoint): a label near a corner of the shared edge
            # must still match that edge, not a perpendicular edge with a nearer midpoint.
            d = _point_seg_dist(tx, ty, pa[0], pa[1], pb[0], pb[1])
            if best is None or d < best[0]:
                best = (d, a, b)
    if best is None:
        return None
    _, a, b = best
    return (a, b, _seg_len(_rel_xy(m1, a), _rel_xy(m1, b)))


def propagate_from_seated(
    m1_b: M1PlotData,
    placement_a: CandidatePlacement,
    m1_a: M1PlotData,
    placed_footprints: list,
) -> CandidatePlacement | None:
    """Seat ``m1_b`` (no parcel/GPS) against the already-placed ``m1_a``.

    Uses the neighbour labels (B names A's survey, or A names B's survey) to find
    the shared edge in each, requires the two measured edge lengths to agree, fits
    the rigid alignment in the orientation that tiles (interiors on opposite sides),
    and rejects if scale is off-band or the result overlaps any ``placed_footprints``.
    Returns a ``CandidatePlacement`` (``method="propagated"``, ``passes_gate=True``)
    or None when no safe placement exists.
    """
    from shapely.geometry import Polygon

    # Resolve the shared edge on BOTH plots (try B-names-A, then A-names-B).
    eb = _neighbor_edge(m1_b, m1_a.survey_number)
    ea = _neighbor_edge(m1_a, m1_b.survey_number)
    if eb is None and ea is None:
        return None

    # We need A's shared-edge corners (to know the absolute target endpoints) and
    # B's shared-edge corners (the source to map). If only one side has a label,
    # fall back to length-matching the other side's edge.
    def _match_by_length(m1, target_len):
        best = None
        for a, b in _ring_edges(m1):
            L = _seg_len(_rel_xy(m1, a), _rel_xy(m1, b))
            if _len_agrees(L, target_len):
                if best is None or abs(L - target_len) < best[0]:
                    best = (abs(L - target_len), a, b, L)
        return None if best is None else (best[1], best[2], best[3])

    if eb is not None and ea is None:
        a_corner_a, a_corner_b, len_a = (None, None, None)
        mb = _match_by_length(m1_a, eb[2])
        if mb is None:
            return None
        a_corner_a, a_corner_b, len_a = mb
        b_corner_a, b_corner_b, len_b = eb
    elif ea is not None and eb is None:
        a_corner_a, a_corner_b, len_a = ea
        mb = _match_by_length(m1_b, ea[2])
        if mb is None:
            return None
        b_corner_a, b_corner_b, len_b = mb
    else:
        a_corner_a, a_corner_b, len_a = ea
        b_corner_a, b_corner_b, len_b = eb

    if not _len_agrees(len_a, len_b):
        return None

    adj_a = placement_a.adjusted
    if not (0 <= a_corner_a < len(adj_a) and 0 <= a_corner_b < len(adj_a)):
        return None
    P1 = np.array([adj_a[a_corner_a][0], adj_a[a_corner_a][1]], float)
    P2 = np.array([adj_a[a_corner_b][0], adj_a[a_corner_b][1]], float)

    Q1 = np.array(_rel_xy(m1_b, b_corner_a), float)
    Q2 = np.array(_rel_xy(m1_b, b_corner_b), float)
    all_b = m1_b.stone_positions()

    a_footprint = placement_a.footprint()
    best = None  # (overlap, adjusted, R, s, t)
    for dst in ((P1, P2), (P2, P1)):       # both endpoint correspondences
        R, s, t, _r = umeyama(np.array([Q1, Q2]), np.array([dst[0], dst[1]]))
        if not (PROP_SCALE_LO <= s <= PROP_SCALE_HI):
            continue
        adjusted = s * (all_b @ R.T) + t
        ring = [i for i in m1_b.outer_stone_indices if 0 <= i < len(adjusted)]
        if len(ring) < 3:
            continue
        poly = Polygon([(float(adjusted[i][0]), float(adjusted[i][1])) for i in ring])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        # Overlap with the seating neighbour A: the two MUST tile (share the edge,
        # not the interior). The correct orientation minimises interior overlap.
        ov_a = 0.0
        if a_footprint is not None and poly.intersects(a_footprint):
            ov_a = poly.intersection(a_footprint).area / max(
                min(poly.area, a_footprint.area), 1e-9)
        if best is None or ov_a < best[0]:
            best = (ov_a, adjusted, R, float(s), t, poly)

    if best is None:
        return None
    ov_a, adjusted, R, s, t, poly = best
    if ov_a > PROP_MAX_OVERLAP:
        return None

    def _passes_overlap_gates(candidate_poly) -> bool:
        if a_footprint is not None and candidate_poly.intersects(a_footprint):
            ov = candidate_poly.intersection(a_footprint).area / max(
                min(candidate_poly.area, a_footprint.area), 1e-9)
            if ov > PROP_MAX_OVERLAP:
                return False
        for fp in placed_footprints:
            if fp is None or not candidate_poly.intersects(fp):
                continue
            ov = candidate_poly.intersection(fp).area / max(
                min(candidate_poly.area, fp.area), 1e-9)
            if ov > PROP_MAX_OVERLAP:
                return False
        return True

    # Reject overlap with ANY already-placed plot (global tiling).
    if not _passes_overlap_gates(poly):
        return None

    # ---- >=3-point least-squares refinement (min 3 stone matches per FMB) ----
    # The 2-point edge fit is EXACT on its two endpoints, so any endpoint error goes
    # straight into the placement. Adjacent parcels share corners beyond the matched
    # edge: after the initial alignment, every other B corner that lands on an A
    # stone (within PROP_REFIT_SNAP_M) is the same physical corner -> an extra
    # correspondence. With >=3 pairs the similarity is re-fit by least squares
    # (errors average out). The re-fit must re-pass scale + tiling gates; otherwise
    # the gated 2-point result stands (never lose a valid placement to refinement).
    n_fit_pts = 2
    a_stones = np.asarray(adj_a, float)
    src_pairs = [Q1, Q2]
    dst_pairs = [P1, P2]
    used_b = {b_corner_a, b_corner_b}
    used_a = {a_corner_a, a_corner_b}
    for bi in m1_b.outer_stone_indices:
        if bi in used_b or not (0 <= bi < len(adjusted)):
            continue
        d = np.hypot(a_stones[:, 0] - adjusted[bi][0],
                     a_stones[:, 1] - adjusted[bi][1])
        ai = int(np.argmin(d))
        if float(d[ai]) <= PROP_REFIT_SNAP_M and ai not in used_a:
            src_pairs.append(np.array(_rel_xy(m1_b, bi), float))
            dst_pairs.append(np.array([a_stones[ai][0], a_stones[ai][1]], float))
            used_b.add(bi)
            used_a.add(ai)
    if len(src_pairs) >= 3:
        R2, s2, t2, _r2 = umeyama(np.array(src_pairs), np.array(dst_pairs))
        if PROP_SCALE_LO <= s2 <= PROP_SCALE_HI:
            adjusted2 = s2 * (all_b @ R2.T) + t2
            ring2 = [i for i in m1_b.outer_stone_indices if 0 <= i < len(adjusted2)]
            if len(ring2) >= 3:
                poly2 = Polygon([(float(adjusted2[i][0]), float(adjusted2[i][1]))
                                 for i in ring2])
                if not poly2.is_valid:
                    poly2 = poly2.buffer(0)
                if (not poly2.is_empty and poly2.area > 0
                        and _passes_overlap_gates(poly2)):
                    adjusted, R, s, t, poly = adjusted2, R2, float(s2), t2, poly2
                    n_fit_pts = len(src_pairs)

    return CandidatePlacement(
        method="propagated",
        R=R, s=s, t=t,
        adjusted=adjusted,
        corner_ring=list(m1_b.outer_stone_indices),
        passes_gate=True,
        scale=s,
        note=f"propagated from seated {m1_a.survey_number} "
             f"(shared edge {len_a:.1f}/{len_b:.1f} m, tile-overlap {ov_a:.0%}, "
             f"{n_fit_pts}-pt fit)",
    )


# Label-free geometric propagation: minimum coincident corners (endpoints + extras)
# required to accept a placement. A chance length-match of two unrelated edges will not
# also put >=1 further corner on a neighbour corner, so this is the 0-FP gate (client's
# ">= 3 stone points per FMB"). GENERAL -- pure geometry, no OCR labels, no per-village value.
GEO_MIN_SHARED_CORNERS = 3


def propagate_geometric(
    m1_b: M1PlotData,
    placement_a: CandidatePlacement,
    m1_a: M1PlotData,
    placed_footprints: list,
    *,
    min_shared: int = GEO_MIN_SHARED_CORNERS,
) -> CandidatePlacement | None:
    """Seat ``m1_b`` against already-placed ``m1_a`` by a SHARED EDGE found purely
    geometrically -- NO neighbour labels (they are too noisy to rely on).

    For every A corner-edge x every B corner-edge of matching length, both endpoint
    orientations are fitted (rigid, scale~1); a candidate is kept only if, after
    alignment, at least ``min_shared`` B corners coincide with A corners (the two edge
    endpoints + >=1 more) AND the plot tiles (no interior overlap with A or any placed
    plot). The extra-corner corroboration is what makes a chance length-match impossible
    to accept -- so this is 0-FP without any label. The winning pose is then re-fit by
    least squares over ALL coincident corners. Returns a ``CandidatePlacement``
    (``method="relative"``) or None.
    """
    from shapely.geometry import Polygon

    adj_a = np.asarray(placement_a.adjusted, float)
    a_ring = [i for i in placement_a.corner_ring if 0 <= i < len(adj_a)]
    if len(a_ring) < 3:
        return None
    a_corners = adj_a[np.array(a_ring)]                    # absolute UTM A corners
    a_edges = [(a_corners[k], a_corners[(k + 1) % len(a_corners)])
               for k in range(len(a_corners))]

    all_b = m1_b.stone_positions()
    b_ring = [i for i in m1_b.outer_stone_indices if 0 <= i < len(all_b)]
    if len(b_ring) < 3:
        return None
    b_corners_rel = all_b[np.array(b_ring)]
    b_edges = [(b_corners_rel[k], b_corners_rel[(k + 1) % len(b_corners_rel)])
               for k in range(len(b_corners_rel))]

    a_fp = placement_a.footprint()

    def _shared_count(adjusted_ring: np.ndarray) -> int:
        """How many B corners land within EDGE_COINCIDE_TOL of a DISTINCT A corner."""
        used, cnt = set(), 0
        for bc in adjusted_ring:
            d = np.hypot(a_corners[:, 0] - bc[0], a_corners[:, 1] - bc[1])
            ai = int(np.argmin(d))
            if d[ai] <= EDGE_COINCIDE_TOL and ai not in used:
                used.add(ai); cnt += 1
        return cnt

    best = None  # (shared_count, -overlap, R, s, t, adjusted, poly)
    for (A1, A2) in a_edges:
        La = float(np.hypot(*(A2 - A1)))
        for (B1, B2) in b_edges:
            Lb = float(np.hypot(*(B2 - B1)))
            if not _len_agrees(La, Lb):
                continue
            for dst in ((A1, A2), (A2, A1)):
                R, s, t, _r = umeyama(np.array([B1, B2]), np.array([dst[0], dst[1]]))
                if not (PROP_SCALE_LO <= s <= PROP_SCALE_HI):
                    continue
                adjusted = s * (all_b @ R.T) + t
                ring_xy = adjusted[np.array(b_ring)]
                poly = Polygon([(float(x), float(y)) for x, y in ring_xy])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty or poly.area <= 0:
                    continue
                # tiling gate vs A and all placed
                ov = 0.0
                if a_fp is not None and poly.intersects(a_fp):
                    ov = poly.intersection(a_fp).area / max(min(poly.area, a_fp.area), 1e-9)
                if ov > PROP_MAX_OVERLAP:
                    continue
                bad = False
                for fp in placed_footprints:
                    if fp is None or not poly.intersects(fp):
                        continue
                    o = poly.intersection(fp).area / max(min(poly.area, fp.area), 1e-9)
                    if o > PROP_MAX_OVERLAP:
                        bad = True; break
                if bad:
                    continue
                shared = _shared_count(ring_xy)
                if shared < min_shared:
                    continue
                key = (shared, -ov)
                if best is None or key > (best[0], best[1]):
                    best = (shared, -ov, R, float(s), t, adjusted, poly)

    if best is None:
        return None
    shared, neg_ov, R, s, t, adjusted, poly = best

    # Least-squares re-fit over ALL coincident corners (>=3) so no single endpoint error
    # drives the placement; must re-pass scale, else the gated 2-point pose stands.
    src, dst = [], []
    used_a = set()
    for bi in b_ring:
        bp = adjusted[bi]
        d = np.hypot(a_corners[:, 0] - bp[0], a_corners[:, 1] - bp[1])
        ai = int(np.argmin(d))
        if d[ai] <= EDGE_COINCIDE_TOL and ai not in used_a:
            src.append(all_b[bi]); dst.append(a_corners[ai]); used_a.add(ai)
    if len(src) >= 3:
        R2, s2, t2, _r2 = umeyama(np.array(src, float), np.array(dst, float))
        if PROP_SCALE_LO <= s2 <= PROP_SCALE_HI:
            adj2 = s2 * (all_b @ R2.T) + t2
            ring2 = adj2[np.array(b_ring)]
            p2 = Polygon([(float(x), float(y)) for x, y in ring2])
            if not p2.is_valid:
                p2 = p2.buffer(0)
            # Re-verify the refit STILL tiles (refit can nudge into an overlap); only adopt
            # it if it does, else the gated pre-refit pose stands.
            ok2 = (not p2.is_empty and p2.area > 0)
            if ok2 and a_fp is not None and p2.intersects(a_fp):
                ok2 = (p2.intersection(a_fp).area / max(min(p2.area, a_fp.area), 1e-9)
                       <= PROP_MAX_OVERLAP)
            if ok2:
                for fp in placed_footprints:
                    if fp is None or not p2.intersects(fp):
                        continue
                    if (p2.intersection(fp).area / max(min(p2.area, fp.area), 1e-9)
                            > PROP_MAX_OVERLAP):
                        ok2 = False
                        break
            if ok2:
                R, s, t, adjusted = R2, float(s2), t2, adj2

    return CandidatePlacement(
        method="relative",
        R=R, s=s, t=t,
        adjusted=adjusted,
        corner_ring=list(m1_b.outer_stone_indices),
        passes_gate=True,
        scale=s,
        note=f"relative stone-match to {m1_a.survey_number} "
             f"({shared} shared corners, label-free)",
    )
