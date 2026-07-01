"""Coordinate transformation: Umeyama similarity + least-squares cadastral adjustment.

Two-stage georeferencing:

1. **Umeyama similarity transform** (SVD-based):
   Initial rough placement. Takes matched (M1_stone -> surveyor_stone) pairs
   and finds the optimal scale, rotation, and translation that maps M1
   coordinates toward surveyor UTM coordinates.

2. **Least-squares cadastral adjustment** (scipy.optimize.least_squares):
   Fine-tuning. Holds surveyor stone positions nearly fixed (field_weight=1000)
   while soft-pulling edge lengths toward FMB-measured values (dist_weight=1.0).
   This distributes the ~0.2m chain-survey discrepancy evenly across all edges.

Reference: Umeyama (1991), "Least-squares estimation of transformation
parameters between two point patterns", IEEE PAMI 13(4).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

_log = logging.getLogger(__name__)


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Compute the optimal similarity transform (R, s, t) mapping src -> dst.

    Uses SVD to find rotation R, uniform scale s, and translation t that
    minimize:
        sum_i  w_i * || s * R @ src[i] + t - dst[i] ||^2

    Parameters
    ----------
    src : (N, 2) source points (M1 stone coordinates)
    dst : (N, 2) destination points (surveyor UTM coordinates)
    weights : (N,) optional per-point weights (higher = more trust this pair)

    Returns
    -------
    R : (2, 2) rotation matrix
    s : float scale factor
    t : (2,) translation vector
    residuals : (N,) per-point residuals after transform
    """
    n = src.shape[0]
    assert dst.shape[0] == n, "src and dst must have same number of points"
    assert src.shape[1] == 2 and dst.shape[1] == 2

    if weights is None:
        weights = np.ones(n)

    # Degenerate-input guard: fewer than 2 points give no defined similarity transform.
    # Return an explicit degenerate result so callers reject cleanly (see below).
    if n < 2:
        return np.eye(2), 0.0, np.zeros(2), np.full(n, float("inf"))

    # Weighted centroids
    w_sum = weights.sum()
    src_centroid = (weights[:, None] * src).sum(axis=0) / w_sum
    dst_centroid = (weights[:, None] * dst).sum(axis=0) / w_sum

    # Center the points
    src_c = src - src_centroid
    dst_c = dst - dst_centroid

    # Zero-variance guard: all source points coincide after de-meaning, so the scale
    # denominator (src_var) below is 0 -- the division would emit a RuntimeWarning and
    # propagate NaN into s / t / residuals. Every caller already rejects a NaN/0 scale via
    # the 0.5 < s < 2.0 band, but the warning clutters logs and NaN residuals are a smell.
    # Return an explicit degenerate result (R=I, s=0, t=0, residuals=+inf) so callers that
    # gate on a finite residual also reject cleanly, with no log noise.
    src_var = (weights[:, None] * src_c ** 2).sum()
    if not np.isfinite(src_var) or src_var < 1e-9:
        return np.eye(2), 0.0, np.zeros(2), np.full(n, float("inf"))

    # Weighted covariance
    H = (src_c * weights[:, None]).T @ dst_c  # (2, 2)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det(R) = +1)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    # Compute scale (src_var computed in the zero-variance guard above)
    s = np.trace(np.diag(S) @ sign_matrix) / src_var

    # Translation
    t = dst_centroid - s * (R @ src_centroid)

    # Compute residuals
    transformed = (s * (src @ R.T) + t)
    residuals = np.sqrt(np.sum((transformed - dst) ** 2, axis=1))

    _log.info("Umeyama: scale=%.6f, rotation=%.2f deg, translation=(%.2f, %.2f)",
              s, math.degrees(math.atan2(R[1, 0], R[0, 0])), t[0], t[1])
    _log.info("Umeyama residuals: mean=%.3fm, max=%.3fm",
              residuals.mean(), residuals.max())

    return R, s, t, residuals


def cadastral_adjust(
    m1_positions: np.ndarray,
    surveyor_positions: np.ndarray,
    matched_pairs: list[tuple[int, int]],
    edge_pairs: list[tuple[int, int, float]],
    field_weight: float = 1000.0,
    dist_weight: float = 1.0,
    umeyama_result: Optional[tuple] = None,
    robust: bool = False,
    f_scale: float = 4.0,
) -> np.ndarray:
    """Least-squares cadastral adjustment.

    Adjusts stone positions to satisfy two competing objectives:
      1. Stay close to surveyor's field-measured positions (high weight)
      2. Preserve FMB-measured edge lengths (lower weight)

    This distributes the chain-survey discrepancy (~0.2m per edge) evenly
    across all edges instead of forcing a rigid transform.

    Parameters
    ----------
    m1_positions : (N, 2) all M1 stone positions (relative coords)
    surveyor_positions : (M, 2) all surveyor stone positions (UTM)
    matched_pairs : [(m1_idx, surveyor_idx), ...] stone correspondences
    edge_pairs : [(m1_idx_a, m1_idx_b, target_length_m), ...] edges to preserve
    field_weight : weight for surveyor position constraints (default 1000)
    dist_weight : weight for edge-length constraints (default 1.0)
    umeyama_result : optional (R, s, t) for initial placement
    robust : when True, use a soft_l1 (smooth Huber) loss instead of plain L2 so a
        single bad correspondence -- a stone snapped to the WRONG neighbour field
        point within tolerance, or an OCR edge-length outlier -- is down-weighted
        instead of dragging every other corner toward it. This closes the held gap
        where an L2 fit pins a corner to a wrong-but-close stone and the residual
        still reads ~0 ("certified-clean mis-snap"). Default False keeps the
        validated INGUR path (plain L2) byte-identical.
    f_scale : soft_l1 transition scale, in the SCALED residual space (only used
        when ``robust=True``). Residuals below ``f_scale`` stay ~quadratic; larger
        ones are progressively down-weighted. With ``field_weight=1000`` a clean
        DGPS field pair (~0.05 m) maps to ~1.6 in scaled space, so ``f_scale`` MUST
        stay above that (default 4.0) or clean field inliers get down-weighted; at
        4.0 it leaves clean field+edge residuals quadratic while rejecting gross
        (>~4 m edge / mismatched-stone) outliers. See module test for the planted
        outlier that L2 certifies clean and soft_l1 rejects.

    Returns
    -------
    adjusted : (N, 2) adjusted stone positions in UTM coordinates
    """
    n_stones = m1_positions.shape[0]

    # Initialize with Umeyama transform if available
    if umeyama_result is not None:
        R, s, t = umeyama_result
        x0 = s * (m1_positions @ R.T) + t
    else:
        x0 = np.zeros_like(m1_positions, dtype=float)
        for m1_idx, surv_idx in matched_pairs:
            x0[m1_idx] = surveyor_positions[surv_idx]
        if matched_pairs:
            avg_offset = np.mean(
                [surveyor_positions[surv] - m1_positions[m1]
                 for m1, surv in matched_pairs],
                axis=0
            )
            for i in range(n_stones):
                if np.allclose(x0[i], 0.0):
                    x0[i] = m1_positions[i] + avg_offset

    x0_flat = x0.flatten()

    def residuals(flat: np.ndarray) -> np.ndarray:
        pos = flat.reshape(-1, 2)
        res = []

        # Field position constraints
        for m1_idx, surv_idx in matched_pairs:
            diff = pos[m1_idx] - surveyor_positions[surv_idx]
            res.append(math.sqrt(field_weight) * diff[0])
            res.append(math.sqrt(field_weight) * diff[1])

        # Edge length constraints
        for idx_a, idx_b, target_len in edge_pairs:
            actual_vec = pos[idx_b] - pos[idx_a]
            actual_len = np.linalg.norm(actual_vec)
            if actual_len < 1e-10:
                actual_len = 1e-10
            err = math.sqrt(dist_weight) * (actual_len - target_len)
            res.append(err)

        return np.array(res)

    result = least_squares(
        residuals, x0_flat,
        method='trf',
        loss=('soft_l1' if robust else 'linear'),
        f_scale=(f_scale if robust else 1.0),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=5000,
    )

    adjusted = result.x.reshape(-1, 2)

    # Pin UNCONSTRAINED stones back to their Umeyama position.
    # A stone that is neither matched (field constraint) nor an endpoint of any
    # edge (length constraint) is a free variable: its Jacobian columns are all
    # zero, so the TRF trust-region step is unbounded for it and the solver can
    # fling it to ~1e8. Those are typically interior subdivision-corner stones
    # on the STONES layer that are not part of the outer-boundary edge set.
    # Their honest position is the rigid Umeyama placement (in-range, sensible),
    # so restore it rather than trust the degenerate LSQ output.
    constrained: set[int] = {m1_idx for m1_idx, _ in matched_pairs}
    for idx_a, idx_b, _ in edge_pairs:
        constrained.add(idx_a)
        constrained.add(idx_b)
    for i in range(n_stones):
        if i not in constrained:
            adjusted[i] = x0[i]

    # Report quality
    field_residuals = []
    dist_residuals = []
    for m1_idx, surv_idx in matched_pairs:
        d = np.linalg.norm(adjusted[m1_idx] - surveyor_positions[surv_idx])
        field_residuals.append(d)
    for idx_a, idx_b, target_len in edge_pairs:
        actual = np.linalg.norm(adjusted[idx_b] - adjusted[idx_a])
        dist_residuals.append(actual - target_len)

    _log.info("Cadastral adjustment converged: %s", result.message)
    if field_residuals:
        _log.info("  Field residuals: mean=%.4fm, max=%.4fm (%d pairs)",
                  np.mean(field_residuals), np.max(field_residuals), len(matched_pairs))
    if dist_residuals:
        _log.info("  Edge length residuals: mean=%.4fm, max=%.4fm (%d edges)",
                  np.mean(dist_residuals), np.max(dist_residuals), len(edge_pairs))

    return adjusted


@dataclass
class SeedQuality:
    """Error propagation for an exactly-determined 2-point (seed) placement.

    A 2-corner seed pins the whole plot from a single baseline, so there is NO
    averaging: a small positional error on either seed point rotates the entire
    template about the baseline. The induced position error grows LINEARLY with
    distance from the baseline, so a SHORT baseline on a LARGE plot is dangerous
    even when both seed points are individually accurate.
    """

    baseline_m: float
    sigma_angle_rad: float
    sigma_scale_rel: float
    template_radius_m: float
    max_induced_error_m: float
    ok: bool
    reason: str = ""


def seed_quality(
    seed_src: np.ndarray,
    seed_dst: np.ndarray,
    template_points: np.ndarray | None = None,
    sigma_point_m: float = 0.10,
    max_induced_error_m: float = 2.0,
    min_baseline_m: float = 5.0,
) -> SeedQuality:
    """Propagate seed-point uncertainty to the worst-case placement error.

    For a 2-point similarity fit (rotation + uniform scale + translation) the
    baseline direction has angular standard deviation ``sigma_angle ≈ √2·σ / L``
    (two endpoints, each contributing σ perpendicular, over baseline length L),
    and the scale has the same relative uncertainty along the baseline. A template
    point at distance ``D`` from the baseline midpoint then inherits a position
    error ``≈ sigma_angle · D`` from the rotation uncertainty alone.

    This is the principled gate the corridor/seed path was missing: it REJECTS a
    seed whose worst-case induced error exceeds ``max_induced_error_m`` (default the
    survey-grade field-residual reject bound) or whose baseline is shorter than
    ``min_baseline_m`` -- both being ways a too-short baseline silently amplifies
    field noise across a big plot. It only ADDS a rejection signal; it never
    upgrades a placement.

    Parameters
    ----------
    seed_src : (2, 2) the two seed points in the source (M1) frame.
    seed_dst : (2, 2) the two matched points in the field (UTM) frame.
    template_points : (N, 2) all template points (M1 frame) to bound the radius;
        when None, the seed baseline itself is used as the extent.
    sigma_point_m : assumed per-point field positional noise (m). DGPS ~0.05 m;
        a conservative default of 0.10 m covers FMB-vs-field stone identification.
    max_induced_error_m : reject above this worst-case induced error.
    min_baseline_m : reject baselines shorter than this outright.
    """
    src = np.asarray(seed_src, dtype=float)
    dst = np.asarray(seed_dst, dtype=float)
    baseline = float(np.linalg.norm(dst[1] - dst[0]))
    if baseline < 1e-9:
        return SeedQuality(0.0, float("inf"), float("inf"), 0.0, float("inf"),
                           False, "degenerate seed (coincident points)")

    sigma_angle = math.sqrt(2.0) * sigma_point_m / baseline      # rad
    sigma_scale_rel = math.sqrt(2.0) * sigma_point_m / baseline  # dimensionless

    # Template radius: max distance of any template point from the seed midpoint,
    # carried to the field frame by the seed's own scale (|dst| / |src|).
    src_baseline = float(np.linalg.norm(src[1] - src[0]))
    scale = baseline / src_baseline if src_baseline > 1e-9 else 1.0
    if template_points is not None and len(template_points) > 0:
        mid_src = 0.5 * (src[0] + src[1])
        radius_src = float(np.max(np.linalg.norm(
            np.asarray(template_points, dtype=float) - mid_src, axis=1)))
    else:
        radius_src = 0.5 * src_baseline
    radius_m = radius_src * scale

    induced = sigma_angle * radius_m  # rotation-driven worst-case far-corner error

    ok = True
    reason = ""
    if baseline < min_baseline_m:
        ok = False
        reason = f"seed baseline {baseline:.2f} m < {min_baseline_m:.1f} m minimum"
    elif induced > max_induced_error_m:
        ok = False
        reason = (f"short-baseline amplification: {sigma_point_m:.2f} m point noise "
                  f"on a {baseline:.1f} m baseline induces ~{induced:.2f} m at the "
                  f"far corner ({radius_m:.0f} m out) > {max_induced_error_m:.1f} m")

    return SeedQuality(baseline, sigma_angle, sigma_scale_rel, radius_m, induced,
                       ok, reason)
