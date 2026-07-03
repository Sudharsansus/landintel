"""Unified RIGID stone-matcher -- ONE engine, any target (the client's approach).

The client's directive, verbatim: "take one stone point as base point, try to rotate
and match all stones; if not, take another stone point; until you match all the FMB.
Do not override the length and dimensions and properties."

That is exactly this module:

  * BASE-STONE SEARCH -- every (FMB stone -> target stone) pairing is tried as the
    translation anchor; nothing depends on labels or ordering.
  * ROTATION SEARCH -- with the base pinned, each distance-compatible second
    correspondence fixes the rotation angle; every candidate angle is scored.
  * SCALE LOCKED TO 1 BY CONSTRUCTION -- the transform is built from an angle and a
    translation only; no scale is ever computed or applied, so FMB edge lengths,
    dimensions and properties are preserved EXACTLY (client rule #2). This is
    stronger than fitting a scale and gating it near 1.
  * TRY UNTIL MATCHED, THEN STOP HONESTLY -- the best pose over all base stones wins
    (most matched stones, then lowest residual). If even the best pose matches too
    few stones, that is reported honestly in ``full`` -- the caller decides what a
    partial match is worth (M2 club-all: place with a low-confidence label; M3
    strict: reject). The matcher never fabricates a pose.
  * CONDITIONAL FULL-MATCH BAR -- "make it 5": full confidence needs
    ``min(full_match_bar, n_src)`` matched stones, keyed on the plot's OWN corner
    count, so a 4-corner plot is judged by all 4 corners and is never silently
    excluded by a flat constant (client rule #1: no overfit, data-keyed conditions).

TARGET-AGNOSTIC: M2 passes the TNGIS parcel's corner points; M3 may pass surveyor
field stones. Same engine, two targets -- no per-stage or per-village logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

_log = logging.getLogger(__name__)

# Import the shared confidence bar so "5" lives in ONE place.
from ..m2_club.disposition_thresholds import FULL_MATCH_STONES  # noqa: E402


@dataclass
class RigidStoneMatch:
    """A rigid (rotation + translation, scale == 1) pose matching src onto target."""
    R: np.ndarray                      # (2,2) rotation
    t: np.ndarray                      # (2,) translation
    matched_pairs: list[tuple[int, int]] = field(default_factory=list)
    n_matched: int = 0
    mean_residual: float = float("inf")
    full: bool = False                 # matched >= min(full_match_bar, n_src)
    n_src: int = 0
    required: int = 0                  # the conditional bar actually applied

    @property
    def s(self) -> float:
        return 1.0                     # rigid by construction, never a fitted value

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, float) @ self.R.T) + self.t


def rigid_procrustes(src: np.ndarray, dst: np.ndarray):
    """Best rotation + translation (NO scale) mapping src -> dst, least squares.

    The scale-locked sibling of umeyama: R from the SVD of the centered covariance
    (det +1 enforced -- reflections rejected), t = dst_c - R @ src_c. Because scale
    is never solved, applying (R, t) preserves every source edge length exactly.
    Returns (R, t, residuals) or (I, 0, +inf) for degenerate input.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    if n < 2 or len(dst) != n:
        return np.eye(2), np.zeros(2), np.full(max(n, 1), float("inf"))
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    H = (src - sc).T @ (dst - dc)
    if not np.all(np.isfinite(H)):
        return np.eye(2), np.zeros(2), np.full(n, float("inf"))
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, np.sign(d) or 1.0]) @ U.T
    t = dc - R @ sc
    res = np.sqrt(((src @ R.T + t - dst) ** 2).sum(axis=1))
    return R, t, res


def _count_inliers(placed: np.ndarray, tree, tol: float) -> tuple[list[tuple[int, int]], float]:
    """Greedy one-to-one matching of placed src stones to DISTINCT target stones
    within tol, nearest pairs first. Returns (pairs, mean_residual)."""
    d, idx = tree.query(placed)
    order = np.argsort(d)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    total = 0.0
    for i in order:
        if d[i] > tol:
            break
        j = int(idx[i])
        if j in used:
            # nearest target taken by a closer stone -> try the next-nearest few
            dd, jj = tree.query(placed[i], k=min(4, tree.n))
            found = False
            for dk, jk in zip(np.atleast_1d(dd), np.atleast_1d(jj)):
                if dk <= tol and int(jk) not in used:
                    j, dcur, found = int(jk), float(dk), True
                    break
            if not found:
                continue
        else:
            dcur = float(d[i])
        used.add(j)
        pairs.append((int(i), j))
        total += dcur
    mean = total / len(pairs) if pairs else float("inf")
    return pairs, mean


def rigid_stone_match(
    src_pts: np.ndarray,
    target_pts: np.ndarray,
    *,
    tol_m: float,
    full_match_bar: int = FULL_MATCH_STONES,
    refine: bool = True,
) -> RigidStoneMatch | None:
    """Find the best rigid pose (rotation + translation, scale locked 1) placing
    ``src_pts`` (FMB stones, metres) onto ``target_pts`` (target stones, UTM metres).

    ``tol_m`` is the stone-coincidence tolerance and MUST be supplied by the caller,
    keyed on its data (e.g. cadastre jitter scaled by parcel size) -- no hidden
    magic default, per the no-overfit rule.

    Returns the best ``RigidStoneMatch`` (which may be partial -- check ``.full``)
    or None when no pose matches even 2 stones (a rigid pose needs the base + one
    rotation-fixing correspondence; fewer is no evidence at all).
    """
    from scipy.spatial import cKDTree

    src = np.asarray(src_pts, float)
    tgt = np.asarray(target_pts, float)
    n, m = len(src), len(tgt)
    if n < 2 or m < 2:
        return None

    tree = cKDTree(tgt)
    required = min(int(full_match_bar), n)

    # Pairwise distances from each target stone to all others (for the rotation-
    # fixing second correspondence): sorted once per anchor for pruning.
    best: RigidStoneMatch | None = None

    for i in range(n):                     # base FMB stone (the client's base point)
        for j in range(m):                 # candidate target seat for it
            # second correspondence fixes the rotation: FMB stone k must land on a
            # target stone l at the SAME distance from the base (rigid => distances
            # are invariant). Distance-compatibility prunes the l candidates.
            for k in range(n):
                if k == i:
                    continue
                d_src = float(np.hypot(*(src[k] - src[i])))
                if d_src < max(tol_m, 1e-9):
                    continue               # degenerate baseline, no angle info
                # targets in the annulus [d_src - tol, d_src + tol] around tgt[j]
                cand = tree.query_ball_point(tgt[j], d_src + tol_m)
                for l in cand:             # noqa: E741 - l is the target index
                    if l == j:
                        continue
                    d_tgt = float(np.hypot(*(tgt[l] - tgt[j])))
                    if abs(d_tgt - d_src) > tol_m:
                        continue
                    # rotation from the two baselines; scale NEVER computed
                    a_src = np.arctan2(src[k, 1] - src[i, 1], src[k, 0] - src[i, 0])
                    a_tgt = np.arctan2(tgt[l, 1] - tgt[j, 1], tgt[l, 0] - tgt[j, 0])
                    ang = a_tgt - a_src
                    c, s_ = np.cos(ang), np.sin(ang)
                    R = np.array([[c, -s_], [s_, c]])
                    t = tgt[j] - R @ src[i]
                    placed = src @ R.T + t
                    pairs, mean_res = _count_inliers(placed, tree, tol_m)
                    if len(pairs) < 2:
                        continue
                    if (best is None
                            or len(pairs) > best.n_matched
                            or (len(pairs) == best.n_matched
                                and mean_res < best.mean_residual)):
                        best = RigidStoneMatch(
                            R=R, t=t, matched_pairs=pairs, n_matched=len(pairs),
                            mean_residual=mean_res, n_src=n, required=required,
                            full=len(pairs) >= required)
                        if best.full and best.mean_residual < tol_m * 0.1:
                            break          # can't beat a full, near-exact match
                if best is not None and best.full and best.mean_residual < tol_m * 0.1:
                    break
            if best is not None and best.full and best.mean_residual < tol_m * 0.1:
                break
        if best is not None and best.full and best.mean_residual < tol_m * 0.1:
            break

    if best is None:
        return None

    # Polish: rotation-only Procrustes over ALL matched pairs (still scale-locked),
    # then recount. Keeps the refined pose only if it matches at least as many
    # stones with a residual no worse -- refinement can never lose evidence.
    if refine and best.n_matched >= 2:
        si = np.array([p[0] for p in best.matched_pairs])
        ti = np.array([p[1] for p in best.matched_pairs])
        R2, t2, _ = rigid_procrustes(src[si], tgt[ti])
        placed2 = src @ R2.T + t2
        pairs2, mean2 = _count_inliers(placed2, tree, tol_m)
        if len(pairs2) > best.n_matched or (
                len(pairs2) == best.n_matched and mean2 <= best.mean_residual):
            best = RigidStoneMatch(
                R=R2, t=t2, matched_pairs=pairs2, n_matched=len(pairs2),
                mean_residual=mean2, n_src=n, required=required,
                full=len(pairs2) >= required)

    _log.debug("rigid_stone_match: %d/%d stones matched (required %d, full=%s, "
               "mean residual %.2f m)", best.n_matched, n, required, best.full,
               best.mean_residual)
    return best
