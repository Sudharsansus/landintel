"""M2 method 2 -- seat one FMB from operator GPS / control-point correspondences.

Two corner stones whose real UTM coordinates the client supplies (field GPS, or
two known cadastral corners) fully determine the 2D similarity (rotation + uniform
scale + translation); the FMB polygon carries the rest of the shape. There is NO
geometric guessing -- the operator supplies the identity -- so this is inherently
0-FP. The only failure mode is a SHORT baseline amplifying point error across a
large plot, which ``transform.seed_quality`` gates (``seed_ok``). Disposition is
ACCEPT_SEEDED when the baseline is adequate, REVIEW otherwise.
"""
from __future__ import annotations

import logging

import numpy as np

from ..m2_georef.extract_m1 import M1PlotData
from ..m2_georef.transform import seed_quality, umeyama
from .placement import CandidatePlacement

_log = logging.getLogger(__name__)

# Worst-case far-corner induced error bound (m) and minimum baseline (m); the
# survey-grade reject bounds reused from the M3 seed path.
MAX_INDUCED_ERROR_M = 2.0
MIN_BASELINE_M = 5.0

# Minimum resolved control points for a FULL-QUALITY seat. With 3+ the similarity
# is a LEAST-SQUARES fit (per-point GPS error averages out); an exact 2-point solve
# passes any point error straight into the placement. Client directive 2026-07-02:
# match a minimum of 3 stone points per FMB. A 2-point seat is still computed but
# is demoted to REVIEW (seed_ok=False), never silently ACCEPTed.
MIN_CONTROL_POINTS = 3


def gps_seat(
    m1: M1PlotData,
    control_points: list[tuple[str, tuple[float, float]]],
) -> CandidatePlacement | None:
    """Place ``m1`` from (corner_label, (utm_x, utm_y)) control points.

    Matches each control label to its STONE on the M1 plot and fits the similarity
    by least squares over ALL resolved points. Quality policy (general, applies to
    every village):
      * >= MIN_CONTROL_POINTS (3) resolved AND baseline + residual gates pass
        -> ``seed_ok=True`` (ACCEPT_SEEDED disposition).
      * exactly 2 resolved -> placement returned but ``seed_ok=False`` (REVIEW):
        a 2-point fit has no redundancy, so a human confirms it.
      * < 2 resolved -> None (cannot determine the similarity at all).
    """
    if not control_points:
        return None

    label_idx: dict[str, int] = {}
    for st in m1.stones:
        label_idx.setdefault(str(st.label).strip(), st.index)

    src_pts, dst_pts = [], []
    for label, xy in control_points:
        key = str(label).strip()
        if key in label_idx:
            i = label_idx[key]
            src_pts.append([m1.stones[i].x, m1.stones[i].y])
            dst_pts.append([float(xy[0]), float(xy[1])])
    if len(src_pts) < 2:
        _log.warning("GPS seat for survey %s: <2 control labels resolved to stones",
                     m1.survey_number)
        return None

    src = np.array(src_pts, dtype=float)
    dst = np.array(dst_pts, dtype=float)
    R, s, t, resid = umeyama(src, dst)

    if not (0.5 < s < 2.0):
        _log.warning("GPS seat for survey %s: implausible scale %.3f -> rejected "
                     "(control points likely mislabeled)", m1.survey_number, s)
        return None

    all_pos = m1.stone_positions()
    adjusted = s * (all_pos @ R.T) + t

    # Baseline check on the FARTHEST-apart control pair (worst-case lever arm),
    # not an arbitrary first two.
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(axis=2)
    bi, bj = np.unravel_index(int(np.argmax(d2)), d2.shape)
    sq = seed_quality(src[[bi, bj]], dst[[bi, bj]], template_points=all_pos,
                      max_induced_error_m=MAX_INDUCED_ERROR_M,
                      min_baseline_m=MIN_BASELINE_M)

    n_pts = len(src_pts)
    max_resid = float(np.max(resid)) if np.all(np.isfinite(resid)) else float("inf")
    if n_pts < MIN_CONTROL_POINTS:
        seed_ok, note = False, (f"only {n_pts} control points resolved "
                                f"(<{MIN_CONTROL_POINTS} minimum) -> REVIEW")
        _log.warning("GPS seat for survey %s: %s", m1.survey_number, note)
    elif max_resid > MAX_INDUCED_ERROR_M:
        # 3+ points give a residual: a control point that disagrees with the fit
        # by more than the survey-grade bound means bad GPS or a mislabel.
        seed_ok, note = False, (f"LSQ residual {max_resid:.2f} m > "
                                f"{MAX_INDUCED_ERROR_M} m -> REVIEW")
        _log.warning("GPS seat for survey %s: %s", m1.survey_number, note)
    else:
        seed_ok, note = sq.ok, ("" if sq.ok else sq.reason)

    return CandidatePlacement(
        method="gps_seed",
        R=R, s=float(s), t=t,
        adjusted=adjusted,
        corner_ring=list(m1.outer_stone_indices),
        passes_gate=seed_ok,
        scale=float(s),
        seed_ok=seed_ok,
        note=note,
    )
