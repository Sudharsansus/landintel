"""Ground-truth STONE-ERROR verification for the clubbed M2 output.

The M2 club seats each FMB on the (raster) cadastre, so it is accurate to whatever the
cadastre is (~10-15 m). When a MANUAL / surveyor reference DXF exists (real boundary
stones as POINT entities, e.g. the "compared dxf" the client builds with their compare
tool), this module measures how far our clubbed corner stones fall from the TRUE stones
and, crucially, SEPARATES two very different failures:

  * PLACEMENT error  -- the FMB SHAPE matches the true stones (a rigid fit lands the
    corresponding corners at sub-tol residual) but the whole plot is shifted/rotated.
    Fixable by a better seat / a stone-match refine; M1 is fine.
  * SHAPE error       -- no rigid fit aligns the corners to the true stones. This is an
    M1 extraction problem (wrong/missing corner) and must be fixed upstream.

This is a VERIFIER: it only reads and reports (0-FP discipline -- it never moves a plot
or invents a coordinate). It is the automated form of the client's "verify the stone
errors" step and the ground-truth QA the cadastre self-check cannot provide.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

DEFAULT_TOL = 2.5        # m: a corner within this of a true stone is a MATCH
MATCH_MEDIAN = 3.0       # m: FMB verdict MATCH if median corner error <= this
SHAPE_MIN_FRAC = 0.45    # rigid fit must align >= this fraction of corners to be "shape OK"


@dataclass
class FmbStoneError:
    survey: str
    n_corners: int
    median_err: float
    mean_err: float
    max_err: float
    pct_within_tol: float
    congruent: bool               # a rigid fit aligns a corner subset to true stones
    congruence_inliers: int
    congruence_residual: float
    verdict: str                  # MATCH | SHIFTED | SHAPE_CHECK | NO_TRUTH

    def line(self) -> str:
        return (f"{self.survey:>6} {self.n_corners:>3}c  "
                f"med={self.median_err:6.1f}m max={self.max_err:6.1f}m  "
                f"<=tol {self.pct_within_tol:3.0f}%  "
                f"fit {self.congruence_inliers:>2}/{self.n_corners:<2} "
                f"res={self.congruence_residual:4.2f}m  {self.verdict}")


@dataclass
class StoneVerifyReport:
    n_truth_stones: int
    tol: float
    rows: list[FmbStoneError] = field(default_factory=list)

    @property
    def n_match(self) -> int:
        return sum(1 for r in self.rows if r.verdict == "MATCH")

    @property
    def n_shape_ok(self) -> int:
        """FMBs whose SHAPE is correct (MATCH or SHIFTED) -> M1 is fine."""
        return sum(1 for r in self.rows if r.verdict in ("MATCH", "SHIFTED"))

    @property
    def median_of_medians(self) -> float:
        v = [r.median_err for r in self.rows if r.n_corners]
        return float(np.median(v)) if v else float("nan")

    def to_text(self) -> str:
        out = [f"STONE-ERROR VERIFY vs {self.n_truth_stones} true stones "
               f"(tol {self.tol:.1f} m)",
               f"  village median-of-medians: {self.median_of_medians:.1f} m",
               f"  MATCH (median<= {MATCH_MEDIAN:.0f} m): {self.n_match}/{len(self.rows)}   "
               f"SHAPE OK (M1 sound, placement off): {self.n_shape_ok}/{len(self.rows)}",
               "  " + "-" * 68]
        for r in sorted(self.rows, key=lambda x: -x.median_err):
            out.append("  " + r.line())
        return "\n".join(out)


def load_truth_stones(dxf_path: str | Path,
                      layers: tuple[str, ...] | None = None) -> np.ndarray:
    """Load real boundary stones (POINT entities) from a manual/surveyor reference DXF.

    ``layers`` restricts to those layer names (default: all POINT entities in modelspace,
    which is what a compare/overlay file carries -- one POINT per surveyed stone).
    """
    import ezdxf
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    pts = []
    for e in msp:
        if e.dxftype() != "POINT":
            continue
        if layers is not None and e.dxf.layer not in layers:
            continue
        loc = e.dxf.location
        pts.append((float(loc.x), float(loc.y)))
    return np.asarray(pts, float) if pts else np.empty((0, 2))


def _rigid_2pt(p1, p2, q1, q2, scale_band=(0.9, 1.1)):
    """Rotation+translation mapping p->q from two correspondences, only if the edge
    lengths match (scale in band == a genuine shape correspondence, not a stretch)."""
    vp, vq = p2 - p1, q2 - q1
    lp = float(np.hypot(*vp))
    if lp < 1e-6:
        return None
    s = float(np.hypot(*vq)) / lp
    if not (scale_band[0] < s < scale_band[1]):
        return None
    th = np.arctan2(vq[1], vq[0]) - np.arctan2(vp[1], vp[0])
    c, sn = np.cos(th), np.sin(th)
    R = np.array([[c, -sn], [sn, c]])
    return R, (q1 - R @ p1)


def _congruence(corners: np.ndarray, truth: np.ndarray, tol: float):
    """Best rigid alignment of the FMB corners onto nearby true stones.

    Returns (inliers, residual). A high inlier fraction at low residual means the FMB
    SHAPE matches the surveyed stones (so any large corner error is placement, not shape).
    """
    from scipy.spatial import cKDTree
    if len(corners) < 2 or len(truth) < 2:
        return 0, float("inf")
    tree = cKDTree(truth)
    tdist = np.hypot(*(truth[:, None, :] - truth[None, :, :]).transpose(2, 0, 1))
    best = (0, float("inf"))
    for i, j in combinations(range(len(corners)), 2):
        d_ij = float(np.hypot(*(corners[i] - corners[j])))
        if d_ij < 8.0:
            continue
        # true-stone pairs at the same separation are candidate correspondences.
        aa, bb = np.where(np.abs(tdist - d_ij) < tol)
        for a, b in zip(aa, bb):
            if a == b:
                continue
            fit = _rigid_2pt(corners[i], corners[j], truth[a], truth[b])
            if fit is None:
                continue
            R, t = fit
            proj = (R @ corners.T).T + t
            dist, _ = tree.query(proj)
            inl = int(np.sum(dist <= tol))
            res = float(np.mean(dist[dist <= tol])) if inl else float("inf")
            if inl > best[0] or (inl == best[0] and res < best[1]):
                best = (inl, res)
    return best


def verify_stones(
    fmb_corners: dict[str, np.ndarray],
    truth_stones: np.ndarray,
    tol: float = DEFAULT_TOL,
    *,
    near_radius: float = 220.0,
) -> StoneVerifyReport:
    """Measure each FMB's clubbed corners against ground-truth stones.

    ``fmb_corners`` : {survey -> (M,2) placed corner-ring UTM positions}.
    ``truth_stones``: (N,2) surveyed stone positions in the SAME CRS.
    """
    from scipy.spatial import cKDTree
    rep = StoneVerifyReport(n_truth_stones=len(truth_stones), tol=tol)
    if len(truth_stones) == 0:
        return rep
    tree = cKDTree(truth_stones)
    for sn, corners in fmb_corners.items():
        corners = np.asarray(corners, float)
        if len(corners) < 3:
            continue
        dist, _ = tree.query(corners)
        med = float(np.median(dist))
        near = truth_stones[np.hypot(*(truth_stones - corners.mean(0)).T) < near_radius]
        inl, res = _congruence(corners, near, tol) if len(near) >= 2 else (0, float("inf"))
        congruent = (inl / len(corners)) >= SHAPE_MIN_FRAC and res <= tol
        if med <= MATCH_MEDIAN:
            verdict = "MATCH"
        elif congruent:
            verdict = "SHIFTED"          # shape correct, placement off (M1 fine)
        else:
            verdict = "SHAPE_CHECK"      # inspect M1 extraction for this survey
        rep.rows.append(FmbStoneError(
            survey=sn, n_corners=len(corners), median_err=med,
            mean_err=float(dist.mean()), max_err=float(dist.max()),
            pct_within_tol=float(100 * np.mean(dist <= tol)),
            congruent=congruent, congruence_inliers=inl,
            congruence_residual=(res if np.isfinite(res) else 99.9),
            verdict=verdict))
    return rep


def verify_clubbed_output(
    clubbed_points_csv: str | Path,
    truth_dxf: str | Path,
    tol: float = DEFAULT_TOL,
    truth_layers: tuple[str, ...] | None = None,
) -> StoneVerifyReport:
    """Convenience agent entry: verify a clubbed_points.csv against a reference DXF."""
    import csv
    from collections import defaultdict
    truth = load_truth_stones(truth_dxf, layers=truth_layers)
    rows: dict[str, list] = defaultdict(list)
    with open(clubbed_points_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["survey_number"]].append((float(r["x_utm"]), float(r["y_utm"])))
    corners = {sn: np.asarray(v, float) for sn, v in rows.items()}
    rep = verify_stones(corners, truth, tol=tol)
    _log.info("stone_verify: %d/%d MATCH, %d/%d SHAPE OK, village median %.1f m",
              rep.n_match, len(rep.rows), rep.n_shape_ok, len(rep.rows),
              rep.median_of_medians)
    return rep
