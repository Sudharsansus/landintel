"""Fingerprint + neighborhood matching between M1 plot and surveyor data.

KEY STRUCTURAL INSIGHT: The surveyor's SITE DATA LINE polylines are OPEN paths
(property boundaries traced along the tower corridor), while M1 FMB boundaries
are CLOSED polygons. So we match:

    M1 N-edge closed polygon  vs  surveyor K-edge open subsequence
    where K = N-1, N, or N+1

3-LAYER MATCHING STRATEGY:
  Layer 1: Survey number text matching (exact or fuzzy numeric match)
  Layer 2: Measurement fingerprint -- edge count + sorted lengths (binned to 0.5m)
  Layer 3: Stone neighborhood distance scoring -- spatial coherence verification

This module handles Layers 2 and 3. Layer 1 is a quick pre-filter in the pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .extract_m1 import M1PlotData
from .extract_surveyor import SurveyorData

_log = logging.getLogger(__name__)

# Round edge lengths to this precision for fingerprinting (metres).
_FINGERPRINT_BIN = 0.5

# Maximum acceptable RMS difference per edge in fingerprint comparison.
_FINGERPRINT_TOLERANCE = 1.5

# Maximum acceptable RMS for neighborhood matching.
_NEIGHBOR_TOLERANCE = 5.0

# Maximum number of fingerprint candidates to verify with neighborhood scoring.
_MAX_CANDIDATES_VERIFY = 30


@dataclass
class MatchResult:
    """Result of matching an M1 plot against surveyor data."""
    stone_map: list[int]      # M1 stone index -> surveyor stone index (-1 = unmatched)
    fingerprint_score: float
    neighborhood_score: float
    combined_score: float
    matched: bool = False
    match_method: str = ""
    n_matched_edges: int = 0
    n_matched_stones: int = 0


def _bin(length: float) -> float:
    """Round length to nearest 0.5m bin for fingerprinting."""
    return round(length / _FINGERPRINT_BIN) * _FINGERPRINT_BIN


def _fingerprint(lengths: list[float]) -> tuple[int, tuple[float, ...]]:
    """Create a fingerprint: (edge_count, sorted_binned_lengths)."""
    return (len(lengths), tuple(sorted(_bin(l) for l in lengths)))


def _fingerprint_rms(fp_a: tuple, fp_b: tuple) -> float:
    """RMS of per-edge differences between two fingerprints."""
    n_a, la = fp_a
    n_b, lb = fp_b
    if n_a != n_b:
        return float("inf")
    if n_a == 0:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(la, lb)) / n_a)


def _extract_polyline_chains(surveyor: SurveyorData) -> list[list[tuple[int, int, float]]]:
    """Extract stone sequences and edge lengths from surveyor polylines.

    Returns list of chains, each chain = [(stone_a, stone_b, length), ...].
    """
    chains = []
    for poly_data in surveyor.polylines:
        chain = []
        stone_idx_list = poly_data.stone_indices
        prev_idx = None
        for i, stone_idx in enumerate(stone_idx_list):
            if prev_idx is not None and stone_idx != prev_idx:
                sa = surveyor.stones[prev_idx]
                sb = surveyor.stones[stone_idx]
                length = math.sqrt((sa.x - sb.x) ** 2 + (sa.y - sb.y) ** 2)
                chain.append((prev_idx, stone_idx, length))
            prev_idx = stone_idx
        if chain:
            chains.append(chain)
    return chains


def _sliding_windows(chain: list[tuple[int, int, float]],
                     window_size: int) -> list[list[tuple[int, int, float]]]:
    """Extract all consecutive windows of size window_size from a chain."""
    if window_size > len(chain):
        return []
    return [chain[i:i + window_size]
            for i in range(len(chain) - window_size + 1)]


def _correspondence_coherence(
    m1_idxs: list[int],
    surv_idxs: list[int],
    m1_pos: np.ndarray,
    surveyor: SurveyorData,
) -> float:
    """Mean abs difference of all pairwise stone distances between two ordered
    correspondences -- a frame-independent congruence score (lower = better).

    This is exactly the quantity the neighborhood gate measures, so minimizing
    it here picks the correspondence the acceptance step will reward.
    """
    n = len(m1_idxs)
    if n < 2:
        return float("inf")
    total = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            md = math.dist(m1_pos[m1_idxs[i]], m1_pos[m1_idxs[j]])
            sa = surveyor.stones[surv_idxs[i]]
            sb = surveyor.stones[surv_idxs[j]]
            sd = math.hypot(sa.x - sb.x, sa.y - sb.y)
            total += abs(md - sd)
            cnt += 1
    return total / cnt if cnt else float("inf")


def _best_correspondence(
    m1: M1PlotData,
    m1_seq: list[int],
    surveyor_stones: list[int],
    surveyor: SurveyorData,
) -> list[int]:
    """Pair an ordered M1 stone sequence to a surveyor stone window.

    The M1 boundary is a CLOSED ring whose traversal start is arbitrary, while
    the surveyor window is an OPEN sub-chain with a fixed start. So the correct
    alignment can be any cyclic ROTATION of the M1 sequence, in either
    DIRECTION. We search all 2L rotations/reflections and keep the one with the
    best pairwise-distance coherence -- this is what fixes plots whose
    fingerprint matches but whose stones were previously mis-aligned (high
    neighborhood score) purely because the two sequences started at different
    corners.
    """
    stone_map = [-1] * m1.n_stones
    k = min(len(m1_seq), len(surveyor_stones))
    if k < 2:
        return stone_map

    m1_pos = m1.stone_positions()
    surv = list(surveyor_stones[:k])

    best_score = float("inf")
    best_cand: list[int] | None = None
    for seq in (list(m1_seq), list(reversed(m1_seq))):
        for r in range(len(seq)):
            rotated = (seq[r:] + seq[:r])[:k]
            score = _correspondence_coherence(rotated, surv, m1_pos, surveyor)
            if score < best_score:
                best_score = score
                best_cand = rotated

    if best_cand is not None:
        for i in range(k):
            stone_map[best_cand[i]] = surv[i]
    return stone_map


def _build_stone_map_ordered(
    m1: M1PlotData,
    surveyor_stones: list[int],
    surveyor: SurveyorData,
) -> list[int]:
    """Map M1 outer-boundary stones to a surveyor window (rotation + direction)."""
    return _best_correspondence(m1, m1.outer_stone_indices, surveyor_stones, surveyor)


def _build_stone_map_skip(
    m1: M1PlotData,
    surveyor_stones: list[int],
    surveyor: SurveyorData,
    skip_edge: int,
) -> list[int]:
    """Map M1 stones to a surveyor window with one M1 edge skipped.

    The skipped edge's trailing stone is removed from the M1 ring; the remaining
    stones are aligned to the surveyor window by the same rotation/direction
    coherence search.
    """
    n_m1 = len(m1.outer_stone_indices)
    if n_m1 < 2:
        return [-1] * m1.n_stones
    skip_b_idx = (skip_edge + 1) % n_m1
    m1_order = [m1.outer_stone_indices[i] for i in range(n_m1) if i != skip_b_idx]
    return _best_correspondence(m1, m1_order, surveyor_stones, surveyor)


def _neighborhood_score(
    m1: M1PlotData,
    stone_map: list[int],
    surveyor: SurveyorData,
    k: int = 2,
) -> float:
    """RMS of neighborhood distance differences for matched stone pairs.

    For each matched pair, compare the distance to other matched stones
    between M1 and surveyor. This verifies spatial coherence.
    """
    m1_pos = m1.stone_positions()
    pairs = [(i, j) for i, j in enumerate(stone_map) if j >= 0]
    if len(pairs) < 3:
        return float("inf")

    diffs = []
    for m1_idx, surv_idx in pairs:
        for other_m1, other_surv in pairs:
            if other_m1 <= m1_idx:
                continue
            m1_d = np.linalg.norm(m1_pos[m1_idx] - m1_pos[other_m1])
            sa = surveyor.stones[surv_idx]
            sb = surveyor.stones[other_surv]
            surv_d = math.sqrt((sa.x - sb.x) ** 2 + (sa.y - sb.y) ** 2)
            diffs.append(abs(m1_d - surv_d))

    if not diffs:
        return float("inf")
    return math.sqrt(sum(d ** 2 for d in diffs) / len(diffs))


def match_by_survey_number(
    m1: M1PlotData,
    surveyor: SurveyorData,
) -> Optional[MatchResult]:
    """Layer 1: Try to match by survey number text.

    The surveyor DXF may have text labels containing survey numbers near
    certain boundary features. This is a fast pre-filter.

    NOTE: The INGUR surveyor DXF does NOT contain survey number labels for
    individual plots (it has corridor approach points AP80-AP84 and village
    names). So Layer 1 typically returns None, and matching falls through
    to Layer 2 (fingerprint).
    """
    # This layer is a placeholder for surveyor DXFs that DO embed survey numbers.
    # For INGUR-style DXFs, we skip directly to fingerprint matching.
    return None


def match_by_fingerprint(
    m1: M1PlotData,
    surveyor: SurveyorData,
    tolerance: float = _FINGERPRINT_TOLERANCE,
) -> list[MatchResult]:
    """Layer 2: Match M1 plot against surveyor using edge-length fingerprints.

    Tries window sizes K = N-1, N, N+1 where N is the M1 outer edge count.
    For each window, extracts sliding subsequences from surveyor polylines
    and compares measurement fingerprints.
    """
    n_edges = len(m1.outer_edges)
    if n_edges < 3 or not surveyor.stones:
        return [_no_match(m1)]

    m1_lengths = [e.length_m for e in m1.outer_edges]
    m1_perim = sum(m1_lengths)
    m1_fp = _fingerprint(m1_lengths)
    _log.info("M1 outer boundary: %d edges, perim=%.1fm, fp=%s",
              n_edges, m1_perim, [round(l, 1) for l in m1_fp[1]])

    # Extract surveyor polyline chains
    chains = _extract_polyline_chains(surveyor)
    _log.info("Surveyor: %d polylines extracted into chains", len(chains))

    # Try window sizes: N-1, N, N+1
    window_sizes = [w for w in [n_edges - 1, n_edges, n_edges + 1] if w >= 2]

    candidates: list[MatchResult] = []
    perim_tol = 0.20  # 20% perimeter tolerance

    for ws in window_sizes:
        for chain in chains:
            windows = _sliding_windows(chain, ws)
            for win in windows:
                win_lengths = [length for _, _, length in win]
                win_perim = sum(win_lengths)

                # Perimeter filter
                if ws == n_edges:
                    if abs(win_perim - m1_perim) / max(m1_perim, 1) > perim_tol:
                        continue
                elif ws == n_edges - 1:
                    possible_perims = [m1_perim - l for l in m1_lengths]
                    best_diff = min(abs(wp - win_perim) for wp in possible_perims)
                    if best_diff / max(m1_perim, 1) > perim_tol:
                        continue
                elif ws == n_edges + 1:
                    if abs(win_perim - m1_perim) / max(m1_perim, 1) > perim_tol:
                        continue

                # Exact N-edge match: direct fingerprint comparison
                if ws == n_edges:
                    win_fp = _fingerprint(win_lengths)
                    fp_dist = _fingerprint_rms(m1_fp, win_fp)
                    if fp_dist > tolerance:
                        continue

                    win_stones = [win[0][0]] + [b for _, b, _ in win]
                    stone_map = _build_stone_map_ordered(m1, win_stones, surveyor)

                    candidates.append(MatchResult(
                        stone_map=stone_map,
                        fingerprint_score=fp_dist,
                        neighborhood_score=float("inf"),
                        combined_score=fp_dist,
                        matched=True,
                        match_method=f"fingerprint_N{ws}",
                        n_matched_edges=ws,
                        n_matched_stones=sum(1 for s in stone_map if s >= 0),
                    ))

                elif ws == n_edges - 1:
                    win_fp = _fingerprint(win_lengths)
                    best_skip = _find_best_edge_skip(m1_lengths, win_fp, tolerance)
                    if best_skip is None:
                        continue

                    skip_idx, fp_dist = best_skip
                    win_stones = [win[0][0]] + [b for _, b, _ in win]
                    stone_map = _build_stone_map_skip(m1, win_stones, surveyor, skip_idx)

                    candidates.append(MatchResult(
                        stone_map=stone_map,
                        fingerprint_score=fp_dist * 1.2,
                        neighborhood_score=float("inf"),
                        combined_score=fp_dist * 1.2,
                        matched=True,
                        match_method=f"fingerprint_N{ws}_skip{skip_idx}",
                        n_matched_edges=ws,
                        n_matched_stones=sum(1 for s in stone_map if s >= 0),
                    ))

    _log.info("Fingerprint: %d candidates found", len(candidates))
    return candidates


def _find_best_edge_skip(
    m1_lengths: list[float],
    target_fp: tuple,
    tolerance: float,
) -> Optional[tuple[int, float]]:
    """Find which M1 edge to skip so remaining edges match target_fp."""
    n = len(m1_lengths)
    best = None
    for skip in range(n):
        remaining = [m1_lengths[i] for i in range(n) if i != skip]
        remaining_fp = _fingerprint(remaining)
        dist = _fingerprint_rms(remaining_fp, target_fp)
        if dist <= tolerance and (best is None or dist < best[1]):
            best = (skip, dist)
    return best


# Inlier distance (metres) for geometric congruence: an M1 corner counts as a
# hit when its transformed position lands within this of a surveyor stone. Set
# above the measured FMB-vs-field discrepancy (~3-5 m) so real plots survive.
_GEOM_INLIER_TOL = 5.0

# A plot is matched on ABSOLUTE congruent-inlier count + residual, NOT fraction.
# Rationale: N distinct surveyor stones landing within inlier_tol of the
# transformed corner template is exponentially unlikely by chance
# (~(rho*pi*tol^2)^(N-2) over the RANSAC trial budget), so the inlier COUNT is
# the real false-positive guard. The FRACTION matched is NOT a reliable
# denominator here for two structural reasons:
#   1. M1 writes ALL corner stones (incl. interior/subdivision corners the
#      surveyor never traced) -- they inflate the corner count but can never be
#      inliers, so frac is biased low for a perfectly good match.
#   2. The surveyor only traces the corridor-crossing SUBSET of each plot's ring,
#      so even the OUTER boundary is only partly covered by field stones.
# We therefore gate on (inliers >= MIN_INLIERS AND mean_res <= inlier_tol), with
# only a low FRAC_FLOOR to reject degenerate tiny-overlap geometry. The
# ACCEPT/REVIEW/REJECT split (pipeline.py) does the final disposition on the
# absolute inlier count.
_GEOM_MIN_INLIERS = 4
_GEOM_MIN_FRAC_FLOOR = 0.30   # degenerate-overlap backstop only (was a 0.70 gate)

# Seat-locality gate (FP guard against wrong-seat congruence). In a dense stone
# cloud a near-congruent N-gon subset exists BY CHANCE far from the plot's true
# position -- measured on INGUR, plots 667/668/669/670/698/699 each found a
# 6-8 inlier subset at ~2 m residual that placed them 1.2-2.6 km from their true
# seat. Inlier count + residual cannot see this (the far subset is just as
# congruent). The INDEPENDENT signal is the plot's expected position (its S3
# cadastral label, keyed by survey number -- a different source than the stone
# geometry the matcher uses): a correct placement's corner-ring centroid lands a
# small fraction of one parcel from its label. INGUR seat distances separate
# cleanly with a ~750 m empty gap -- legitimate matches {25,101,124,165,378} m,
# wrong seats {1128,...,2554} m -- so 600 m sits safely in the gap (>220 m above
# the largest true match, >520 m below the smallest wrong seat). The gate can
# ONLY flip matched True->False, never create a match, so it cannot produce a
# false positive; when no expected position is supplied it is a no-op.
_GEOM_MAX_SEAT_DIST = 600.0


def _similarity_from_pair(
    a: np.ndarray, b: np.ndarray, p: np.ndarray, q: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Exact similarity (R, s, t) mapping segment a->b onto p->q."""
    va, vq = b - a, q - p
    la, lq = float(np.hypot(*va)), float(np.hypot(*vq))
    if la < 1e-9 or lq < 1e-9:
        return np.eye(2), 1.0, p - a
    s = lq / la
    ang = math.atan2(vq[1], vq[0]) - math.atan2(va[1], va[0])
    c, sn = math.cos(ang), math.sin(ang)
    R = np.array([[c, -sn], [sn, c]])
    t = p - s * (R @ a)
    return R, s, t


def geometric_match(
    m1: M1PlotData,
    surveyor: SurveyorData,
    inlier_tol: float = _GEOM_INLIER_TOL,
    allowed_stones: "np.ndarray | None" = None,
    expected_xy: "tuple[float, float] | np.ndarray | None" = None,
    max_seat_dist: float = _GEOM_MAX_SEAT_DIST,
    candidate_sink=None,
) -> MatchResult:
    """Find the M1 corner polygon inside the surveyor stone cloud by congruence.

    For each M1 ring edge (a robust baseline of known length), find every
    surveyor stone pair at a similar distance, fit the exact 2-point similarity
    transform, apply it to ALL M1 corners, and count how many land within
    ``inlier_tol`` of some surveyor stone. The transform with the most inliers
    (tie-break: lowest mean residual) wins; its inlier correspondences become the
    stone map. This is RANSAC over similarity transforms -- it uses the full 2D
    shape, tolerates the ~4 m FMB-vs-field discrepancy, and rejects collisions
    (a wrong place has no congruent N-gon).

    ``allowed_stones`` : optional boolean mask (len == number of surveyor stones).
    When given, both the baseline anchor pairs AND the inlier hits are restricted
    to True stones -- i.e. the plot is matched ONLY within a corridor WINDOW. The
    second-pass propagation uses this to constrain each REVIEW plot to its correct
    schedule neighbourhood, so a near-congruent shape can no longer grab a wrong
    seat elsewhere on the corridor.

    ``expected_xy`` : optional independent expected position for this plot (its S3
    cadastral label point, keyed by survey number). When given, the winning
    transform is REJECTED (matched=False) if the placed corner-ring centroid lands
    farther than ``max_seat_dist`` from it -- the seat-locality FP guard against a
    chance-congruent subset far from the true seat (see _GEOM_MAX_SEAT_DIST). This
    only ever turns a match OFF, so it cannot create a false positive; with no
    ``expected_xy`` it is a no-op.
    """
    corners = m1.outer_stone_indices
    n = len(corners)
    if n < 3 or not surveyor.stones:
        return _no_match(m1)

    if surveyor._stone_tree is None:
        surveyor.build_index()
    tree = surveyor._stone_tree
    surv = surveyor.stone_positions  # (M, 2) property

    m1_pos = m1.stone_positions()
    cpts = np.array([m1_pos[i] for i in corners])  # corner template (N, 2)

    # Ring sanity guard -- M1-QUALITY OBSERVABILITY (client rule: M1 first). A
    # duplicate corner vertex means the M1 extraction handed us a degraded ring;
    # surface it in the log so the fix happens UPSTREAM in M1, never by loosening
    # a gate here. Matching only refuses when the ring is truly degenerate
    # (< 3 DISTINCT corners -- no 2D congruence is defined for a line/point).
    n_distinct = len({(round(float(x), 6), round(float(y), 6)) for x, y in cpts})
    if n_distinct < len(cpts):
        _log.warning(
            "plot %s: corner ring has duplicate vertices (%d corners, %d distinct)"
            " -- M1 extraction fidelity issue; fix upstream in M1",
            m1.survey_number, len(cpts), n_distinct)
    if n_distinct < 3:
        _log.warning("plot %s: degenerate corner ring (< 3 distinct corners) -- no match",
                     m1.survey_number)
        return _no_match(m1)

    # Candidate surveyor pairs come from BOTH traced chain edges and, for each
    # baseline length, any stone pair near that distance (radius query) -- so a
    # plot still matches even if the surveyor never traced that exact edge.
    surv_xy = surv

    best = None  # (n_inliers, -mean_res, stone_map, R, s, t)
    for bi in range(n):
        A = cpts[bi]
        B = cpts[(bi + 1) % n]
        L = float(np.hypot(*(B - A)))
        if L < 1.0:
            continue
        tol_len = max(2.0 * inlier_tol, 0.15 * L)
        # Stones within L+tol of an arbitrary anchor: do a ball query per stone
        # is O(M^2); instead query pairs via the KD-tree at radius L.
        pair_idx = tree.query_ball_point(surv_xy, r=L + tol_len)
        for pi, neighbours in enumerate(pair_idx):
            if allowed_stones is not None and not allowed_stones[pi]:
                continue
            P = surv_xy[pi]
            for qi in neighbours:
                if qi <= pi:
                    continue
                if allowed_stones is not None and not allowed_stones[qi]:
                    continue
                Q = surv_xy[qi]
                d_pq = float(np.hypot(*(Q - P)))
                if abs(d_pq - L) > tol_len:
                    continue
                for p1, p2 in ((P, Q), (Q, P)):
                    R, s, t = _similarity_from_pair(A, B, p1, p2)
                    if not (0.5 < s < 2.0):
                        continue
                    tc = s * (cpts @ R.T) + t
                    dist, idx = tree.query(tc)
                    inmask = dist <= inlier_tol
                    if allowed_stones is not None:
                        inmask = inmask & allowed_stones[idx]
                    ninl = int(inmask.sum())
                    if ninl < _GEOM_MIN_INLIERS:
                        continue
                    # Require distinct surveyor stones (no two corners on one).
                    hit_idx = idx[inmask]
                    if len(set(hit_idx.tolist())) < ninl:
                        continue
                    mean_res = float(dist[inmask].mean())
                    key = (ninl, -mean_res)
                    make_smap = (candidate_sink is not None
                                 or best is None or key > (best[0], -best[1]))
                    if make_smap:
                        smap = [-1] * m1.n_stones
                        for k, cidx in enumerate(corners):
                            if inmask[k]:
                                smap[cidx] = int(idx[k])
                    if candidate_sink is not None:
                        # Coverage-aware selection hook: offer EVERY viable pose (not just the
                        # max-inlier one) to the caller. Default None -> INGUR path byte-identical.
                        candidate_sink(ninl, mean_res, list(smap), R, s, t)
                    if best is None or key > (best[0], -best[1]):
                        best = (ninl, mean_res, smap, R, s, t)

    if best is None:
        return _no_match(m1)

    ninl, mean_res, smap, _R, _s, _t = best
    frac = ninl / n
    # Inlier count + tight residual are the decision; frac only blocks
    # degenerate tiny-overlap geometry (see _GEOM_MIN_FRAC_FLOOR rationale).
    matched = (ninl >= _GEOM_MIN_INLIERS
               and mean_res <= inlier_tol
               and frac >= _GEOM_MIN_FRAC_FLOOR)
    # Seat-locality FP guard: reject a congruent subset that places the plot far
    # from its independent expected position (see _GEOM_MAX_SEAT_DIST). The placed
    # corner-ring centroid is the same point the disposition uses as the footprint
    # centroid. Only ever turns matched off -> cannot create a false positive.
    if matched and expected_xy is not None and max_seat_dist is not None:
        placed = _s * (cpts @ _R.T) + _t
        cx, cy = placed.mean(axis=0)
        seat_dist = math.hypot(cx - expected_xy[0], cy - expected_xy[1])
        if seat_dist > max_seat_dist:
            matched = False
    return MatchResult(
        stone_map=smap,
        fingerprint_score=mean_res,
        neighborhood_score=mean_res,
        combined_score=mean_res,
        matched=matched,
        match_method=f"geometric_{ninl}/{n}_inliers",
        n_matched_edges=ninl,
        n_matched_stones=ninl,
    )


def match_plot(
    m1: M1PlotData,
    surveyor: SurveyorData,
    expected_xy: "tuple[float, float] | np.ndarray | None" = None,
) -> MatchResult:
    """High-level matching: geometric congruence (primary), fingerprint fallback.

    Geometric congruence (RANSAC over 2-point similarity transforms on the
    corner polygon) is the decision-maker -- it uses full 2D shape and rejects
    the edge-length collisions that plagued pure fingerprinting. The edge-length
    fingerprint is retained only as a fallback when geometry is inconclusive.

    ``expected_xy`` : optional independent expected position (the plot's S3
    cadastral label); passed to the seat-locality FP guard in ``geometric_match``
    so a chance-congruent subset far from the true seat is rejected.
    """
    geo = geometric_match(m1, surveyor, expected_xy=expected_xy)
    if geo.matched:
        return geo

    # Fallback: fingerprint (kept for parity / borderline cases). It does not
    # override a geometric rejection -- it only offers a candidate when geometry
    # found nothing.
    results = match_by_fingerprint(m1, surveyor)
    if not results:
        return geo
    results.sort(key=lambda r: r.fingerprint_score)
    for c in results[:_MAX_CANDIDATES_VERIFY]:
        c.neighborhood_score = _neighborhood_score(m1, c.stone_map, surveyor)
        c.combined_score = c.fingerprint_score + 0.5 * c.neighborhood_score
    results.sort(key=lambda r: r.combined_score)
    best = results[0]
    # Fingerprint may only CONFIRM (never override geometry): require a tight
    # neighborhood agreement to accept, otherwise stay unmatched.
    best.matched = (best.fingerprint_score < 2.0 and best.neighborhood_score < 8.0)
    return best if best.matched else geo


def _no_match(m1: M1PlotData) -> MatchResult:
    return MatchResult(
        stone_map=[-1] * m1.n_stones,
        fingerprint_score=float("inf"),
        neighborhood_score=float("inf"),
        combined_score=float("inf"),
        matched=False,
        match_method="none",
    )
