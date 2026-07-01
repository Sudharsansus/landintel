"""Place a plot by registering its boundary EDGES onto the surveyor's traced lines.

WHY this exists (the last automatic lever on corridor data only): corner-stone
RANSAC (`match.geometric_match`) places a plot only when >=4 of its CORNER stones
are exposed in the surveyor cloud. The corridor CLIPS most plots -- it traces only
the strip it crosses -- so a clipped plot may expose <4 corners yet still share a
full BOUNDARY EDGE with the surveyor's traced `SITE DATA LINE` (the actually-walked
cadastral boundary). This module anchors on that edge instead of on corners:

    for each M1 boundary edge (A->B, known length)            [the template]
      for each traced segment (P->Q) of a similar length      [the field truth]
        fit the exact 2-point similarity A->P, B->Q (both directions),
        apply it to the whole plot, and SCORE BY CHAIN COVERAGE
        (fraction of the placed boundary lying on traced lines).

The transform with the HIGHEST chain coverage wins. Chain coverage is the same
signal proven on INGUR to separate true placements (58-100%) from coincidental
ones (12-43%), so MAXIMISING it -- not corner-inlier count -- is what pulls a
clipped plot onto its true traced boundary. Caller gates the result on
coverage >= the ACCEPT bar, schedule identity, schedule-window position, and
non-overlap, so this cannot manufacture a false positive: it only ACCEPTs a plot
whose boundary measurably lies on the surveyor's traced cadastral line.

Uses ONLY the surveyor field DXF (stones + SITE DATA LINE). No external data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .extract_m1 import M1PlotData
from .extract_surveyor import SurveyorData
from .match import _similarity_from_pair
from .transform import cadastral_adjust
from .verify import build_traced_buffer, chain_coverage

_log = logging.getLogger(__name__)

# Inlier tolerance (m) to snap a placed corner to a surveyor stone for refinement.
_SNAP_TOL = 5.0
# A traced segment / M1 edge shorter than this is too ambiguous to anchor on.
_MIN_SEG_LEN = 4.0
# Field/edge weights for the cadastral refinement (same as the main pipeline).
_FIELD_WEIGHT = 1000.0
_DIST_WEIGHT = 1.0


@dataclass
class EdgeRegResult:
    """A boundary-edge registration of one plot onto the traced lines."""
    R: np.ndarray
    s: float
    t: np.ndarray
    adjusted: np.ndarray          # (N,2) UTM positions for all M1 stones
    coverage: float               # chain coverage of the placed boundary
    matched_pairs: list[tuple[int, int]]   # (m1_stone, surveyor_stone) snapped
    n_anchor_corners: int         # corners landing on a surveyor stone


def traced_segments(
    surveyor: SurveyorData,
    keep: "np.ndarray | None" = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """All consecutive vertex pairs of the SITE DATA LINE polylines as segments.

    ``keep`` : optional boolean mask over surveyor stones defining a corridor
    WINDOW; a segment is kept only if BOTH its snapped endpoint stones are in the
    window (restricts registration to the plot's schedule neighbourhood).
    """
    segs: list[tuple[np.ndarray, np.ndarray, float]] = []
    for pl in surveyor.polylines:
        pts = pl.raw_points
        sidx = pl.stone_indices
        for i in range(len(pts) - 1):
            if keep is not None:
                ia = sidx[i] if i < len(sidx) else -1
                ib = sidx[i + 1] if i + 1 < len(sidx) else -1
                if ia < 0 or ib < 0 or not (keep[ia] and keep[ib]):
                    continue
            a = np.array(pts[i], dtype=float)
            b = np.array(pts[i + 1], dtype=float)
            L = float(np.hypot(*(b - a)))
            if L >= _MIN_SEG_LEN:
                segs.append((a, b, L))
    return segs


def register_plot_on_traced(
    m1: M1PlotData,
    surveyor: SurveyorData,
    segments: list[tuple[np.ndarray, np.ndarray, float]] | None = None,
    snap_tol: float = _SNAP_TOL,
    len_tol_frac: float = 0.15,
) -> EdgeRegResult | None:
    """Best similarity placing the M1 plot's boundary onto the traced lines.

    Returns the transform (and cadastral-refined positions) that MAXIMISES chain
    coverage over all (M1 edge -> traced segment) anchor fits, or None if the plot
    has no usable ring / no candidate segment. Does NOT decide ACCEPT/REVIEW --
    the caller gates on the returned ``coverage`` plus identity/position/overlap.
    """
    corners = m1.outer_stone_indices
    n = len(corners)
    if n < 3 or not surveyor.stones:
        return None
    if surveyor._stone_tree is None:
        surveyor.build_index()
    if segments is None:
        segments = traced_segments(surveyor)
    if not segments:
        return None

    m1_pos = m1.stone_positions()
    cpts = np.array([m1_pos[i] for i in corners])           # (N,2) corner ring
    surv_polys = [pl.raw_points for pl in surveyor.polylines]
    # Pre-build the buffered traced lines ONCE and reuse for every candidate fit
    # (chain_coverage otherwise rebuilds it per call -- the search bottleneck).
    traced_buf = build_traced_buffer(surv_polys)
    tree = surveyor._stone_tree

    # M1 corner-to-corner boundary edges (the anchoring templates).
    edges: list[tuple[np.ndarray, np.ndarray, float]] = []
    for i in range(n):
        A = cpts[i]
        B = cpts[(i + 1) % n]
        L = float(np.hypot(*(B - A)))
        if L >= _MIN_SEG_LEN:
            edges.append((A, B, L))
    if not edges:
        return None

    best: tuple | None = None   # (coverage, R, s, t)
    for A, B, L in edges:
        tol_len = max(2.0 * snap_tol, len_tol_frac * L)
        for P, Q, Lt in segments:
            if abs(L - Lt) > tol_len:
                continue
            for p1, p2 in ((P, Q), (Q, P)):
                R, s, t = _similarity_from_pair(A, B, p1, p2)
                if not (0.5 < s < 2.0):
                    continue
                tc = s * (cpts @ R.T) + t
                ring = [(float(tc[k][0]), float(tc[k][1])) for k in range(n)]
                ring_segs = [(ring[k], ring[(k + 1) % n]) for k in range(n)]
                cov = chain_coverage(ring_segs, prepared=traced_buf)
                if best is None or cov > best[0]:
                    best = (cov, R, s, t)

    if best is None:
        return None

    cov, R, s, t = best
    full = s * (m1_pos @ R.T) + t
    # Snap placed corners (and any stone) to nearby surveyor stones, refine.
    dist, idx = tree.query(full)
    matched_pairs = [(i, int(idx[i])) for i in range(len(full)) if dist[i] <= snap_tol]
    n_anchor = sum(1 for i, _ in matched_pairs if i in set(corners))

    adjusted = full
    if len(matched_pairs) >= 2:
        edge_pairs = [(e.stone_a, e.stone_b, e.length_m) for e in m1.outer_edges]
        adjusted = cadastral_adjust(
            m1_positions=m1_pos,
            surveyor_positions=surveyor.stone_positions,
            matched_pairs=matched_pairs,
            edge_pairs=edge_pairs,
            field_weight=_FIELD_WEIGHT,
            dist_weight=_DIST_WEIGHT,
            umeyama_result=(R, s, t),
        )
        ring = [(float(adjusted[i][0]), float(adjusted[i][1])) for i in corners]
        ring_segs = [(ring[k], ring[(k + 1) % n]) for k in range(n)]
        cov = chain_coverage(ring_segs, prepared=traced_buf)

    return EdgeRegResult(
        R=R, s=s, t=t, adjusted=adjusted, coverage=cov,
        matched_pairs=matched_pairs, n_anchor_corners=n_anchor,
    )
