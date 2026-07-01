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
        mid = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
        for tx, ty in targets:
            d = math.hypot(mid[0] - tx, mid[1] - ty)
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
    # Reject overlap with ANY already-placed plot (global tiling).
    for fp in placed_footprints:
        if fp is None or not poly.intersects(fp):
            continue
        ov = poly.intersection(fp).area / max(min(poly.area, fp.area), 1e-9)
        if ov > PROP_MAX_OVERLAP:
            return None

    return CandidatePlacement(
        method="propagated",
        R=R, s=s, t=t,
        adjusted=adjusted,
        corner_ring=list(m1_b.outer_stone_indices),
        passes_gate=True,
        scale=s,
        note=f"propagated from seated {m1_a.survey_number} "
             f"(shared edge {len_a:.1f}/{len_b:.1f} m, tile-overlap {ov_a:.0%})",
    )
