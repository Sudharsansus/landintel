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


def gps_seat(
    m1: M1PlotData,
    control_points: list[tuple[str, tuple[float, float]]],
) -> CandidatePlacement | None:
    """Place ``m1`` from >=2 (corner_label, (utm_x, utm_y)) control points.

    Matches each control label to its STONE on the M1 plot, fits the 2-point (or
    least-squares for >2) similarity, and returns a ``CandidatePlacement``
    (``method="gps_seed"``). ``passes_gate``/``seed_ok`` reflect the seed-quality
    baseline check. Returns None if fewer than two labels resolve to stones.
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
    R, s, t, _resid = umeyama(src, dst)

    if not (0.5 < s < 2.0):
        _log.warning("GPS seat for survey %s: implausible scale %.3f -> rejected "
                     "(control points likely mislabeled)", m1.survey_number, s)
        return None

    all_pos = m1.stone_positions()
    adjusted = s * (all_pos @ R.T) + t

    sq = seed_quality(src[:2], dst[:2], template_points=all_pos,
                      max_induced_error_m=MAX_INDUCED_ERROR_M,
                      min_baseline_m=MIN_BASELINE_M)

    return CandidatePlacement(
        method="gps_seed",
        R=R, s=float(s), t=t,
        adjusted=adjusted,
        corner_ring=list(m1.outer_stone_indices),
        passes_gate=sq.ok,
        scale=float(s),
        seed_ok=sq.ok,
        note="" if sq.ok else sq.reason,
    )
