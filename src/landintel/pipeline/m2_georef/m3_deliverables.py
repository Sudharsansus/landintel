"""M3 stone-cloud finishing: scale-locked placement + honest disposition + deliverables.

This is the FINISHING layer for the stone-cloud M3 path (``run_m3.py``): once the
matcher has paired an M1 plot's corners to surveyor stones, this module

  1. builds the placement with a SCALE-LOCKED rigid fit (rule 2: rotation + translation
     only, scale == 1 by construction via ``rigid_procrustes``). The similarity scale
     ``umeyama`` would have fitted is recorded as a DIAGNOSTIC (``s_fitted``) only -- a
     value far from 1 flags an upstream M1 unit bug, never something to warp,
  2. classifies each plot into a FIRST-CLASS disposition -- ACCEPT / REVIEW / NEEDS_GPS
     / UNMEASURED -- distinguishing "the surveyor never measured near here" (UNMEASURED)
     from "stones are here but the fit is too weak" (NEEDS_GPS); neither is silently
     folded into the other,
  3. writes the three deliverables the client waits on: a georeferenced DXF, a QA
     overlay PNG (plots on the stone cloud), and a per-plot JSON report.

NO per-village constants. The stone bar reuses the client's ``FULL_MATCH_STONES`` (the
"make it 5", applied as ``min(5, n_corners)`` so a plot is judged by its OWN corners);
the ACCEPT residual is the INGUR-validated survey-grade bound, a general quality bar,
not a per-village number. The similarity-scale sanity band is the VALIDATED 0.5-2.0
(widening it was rejected; tightening it risks rejecting valid plots on small M1 noise).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from ..m2_club.disposition_thresholds import CAD_MIN_STONES, FULL_MATCH_STONES
from .stone_matcher import rigid_procrustes
from .transform import umeyama

_log = logging.getLogger(__name__)

# --- M3 survey-grade disposition bounds (general quality bars, not per-village) ---
# ACCEPT residual is the INGUR-validated survey-grade field gate (~2 m); the max is the
# per-pair sanity ceiling ("residual < 3 m" from the validated M3 spec). REVIEW is a
# looser band for "located but confirm". These gate the MEDIAN per-pair distance (robust
# to a single jittered corner), sanity-checked by the MAX. They are NOT tightened to the
# un-sourced 1.0 m an external spec proposed -- 2 m is what real data validated.
M3_ACCEPT_RESIDUAL_MEDIAN_M: float = 2.0
M3_ACCEPT_RESIDUAL_MAX_M: float = 3.0
M3_REVIEW_RESIDUAL_MEDIAN_M: float = 3.5
# Similarity-scale sanity band for the s_fitted DIAGNOSTIC. The validated M2 band; a true
# rigid match onto UTM surveyor stones has s ~ 1. Outside this = an M1 unit bug upstream.
M3_SCALE_LO: float = 0.5
M3_SCALE_HI: float = 2.0
# Cadastre corroboration tolerance. The cadastre seat is a LOCATION PRIOR M3 computes itself by
# running the shared M2 club algorithm on an independently-built TNGIS source (NOT M2's output
# files). Used two ways, both 0-FP: (a) to CROP the surveyor-stone search to the plot's
# neighbourhood (run_m3), and (b) here as an INDEPENDENT cross-check -- a survey-grade stone
# placement within this many metres of the seat is confirmed by TWO independent sources (field
# stones + government cadastre), one that DISAGREES with an available seat is demoted
# ACCEPT->REVIEW. VALUE FROM MEASUREMENT (not a guess): the four true survey-grade placements
# land 10-44 m from the independently-reproduced seat (the seat itself carries the cadastre's
# own ~10-40 m reproduction offset), while chance-congruent decoys relocate ~700 m -- so 60 m
# sits in a ~12x gap, NOT knife-edge / not overfit. The cadastre NEVER sets a coordinate
# (rule 2, crop-only): a wrong seat can only withhold trust, never grant it.
M3_CORROB_TOL_M: float = 60.0

_CONFIDENT = ("ACCEPT",)


@dataclass
class M3Placement:
    """One plot's scale-locked placement onto the surveyor stones + its disposition."""
    survey_number: str
    disposition: str                       # ACCEPT | REVIEW | NEEDS_GPS | UNMEASURED
    R: np.ndarray | None = None            # (2,2) rotation (None if never placed)
    t: np.ndarray | None = None            # (2,) translation
    s_fitted: float = float("nan")         # DIAGNOSTIC similarity scale (placement is s=1)
    ring_utm: np.ndarray | None = None     # (K,2) placed outer corner ring, UTM metres
    n_matched: int = 0
    n_corners: int = 0
    median_residual_m: float = float("nan")
    max_residual_m: float = float("nan")
    cadastre_corrob_m: float = float("nan")   # dist placement<->cadastre seat (NaN = no seat)
    note: str = ""
    m1_file: str = ""
    village: str = ""                          # source village (for the combined district output)

    def footprint(self) -> Polygon | None:
        if self.ring_utm is None or len(self.ring_utm) < 3:
            return None
        p = Polygon([(float(x), float(y)) for x, y in self.ring_utm])
        if not p.is_valid:
            p = p.buffer(0)
        return p if (not p.is_empty and p.area > 0) else None

    # Alias so the runtime agent layer (which speaks ``recommendation``) can read AND
    # demote-write this placement: the VerificationAgent may only turn a confident plot ->
    # REVIEW, and that reflects straight onto ``disposition`` (never a promotion).
    @property
    def recommendation(self) -> str:
        return self.disposition

    @recommendation.setter
    def recommendation(self, value: str) -> None:
        self.disposition = value


def place_scale_locked(src_corners: np.ndarray, dst_stones: np.ndarray):
    """Scale-locked rigid placement of matched M1 corners onto surveyor stones (rule 2).

    ``src_corners`` and ``dst_stones`` are the matched pairs (same length, order-aligned).
    Returns ``(R, t, s_fitted, residuals)`` where (R, t) is the SCALE-1 rigid fit used for
    placement and ``s_fitted`` is the similarity scale ``umeyama`` would have used -- a
    DIAGNOSTIC only. Residuals are the per-pair distances under the emitted (scale-1) fit.
    """
    src = np.asarray(src_corners, float)
    dst = np.asarray(dst_stones, float)
    R, t, residuals = rigid_procrustes(src, dst)          # scale locked to 1
    try:
        _R2, s_fitted, _t2, _ = umeyama(src, dst)         # scale is a diagnostic ONLY
    except Exception:  # noqa: BLE001
        s_fitted = float("nan")
    return R, t, float(s_fitted), residuals


def classify(n_matched: int, n_corners: int, median_resid: float, max_resid: float,
             s_fitted: float, tiles: bool, window_has_stones: bool,
             cadastre_corrob_m: float | None = None) -> tuple[str, str]:
    """First-class M3 disposition cascade (best evidence first). No silent demotions.

    UNMEASURED   -- the surveyor set no stones in this plot's search window (data gap).
    ACCEPT       -- >= min(FULL_MATCH_STONES, n_corners) matched, median residual within
                    survey-grade, max within the sanity ceiling, tiles, scale sane.
    REVIEW       -- located but weaker (>= min(CAD_MIN_STONES, n) matched, looser residual).
    NEEDS_GPS    -- stones are present but no fit clears REVIEW (needs a GPS/operator seed).

    ``cadastre_corrob_m`` (optional) is the distance between the stone placement and the
    INDEPENDENT cadastre seat (survey# -> parcel centroid). When present it is an extra 0-FP
    cross-check applied to an ACCEPT only: within ``M3_CORROB_TOL_M`` -> confirmed by two
    independent sources (kept ACCEPT, noted); beyond it -> the survey-grade fit disagrees with
    where the cadastre says this survey number is, so demote ACCEPT->REVIEW. It can only
    withhold an ACCEPT, never create one (the cadastre never sets a coordinate; rule 2).
    """
    if not window_has_stones:
        return "UNMEASURED", "no surveyor stones in the plot's search window (data gap)"

    accept_bar = min(FULL_MATCH_STONES, n_corners)        # the client's "make it 5", data-keyed
    review_bar = min(CAD_MIN_STONES, n_corners)
    scale_ok = M3_SCALE_LO < s_fitted < M3_SCALE_HI if s_fitted == s_fitted else False

    if (n_matched >= accept_bar
            and median_resid <= M3_ACCEPT_RESIDUAL_MEDIAN_M
            and max_resid <= M3_ACCEPT_RESIDUAL_MAX_M
            and scale_ok and tiles):
        reason = (f"n={n_matched}/{n_corners} med={median_resid:.2f}m "
                  f"max={max_resid:.2f}m s_fit={s_fitted:.3f} (survey-grade)")
        if cadastre_corrob_m is not None and cadastre_corrob_m == cadastre_corrob_m:
            if cadastre_corrob_m <= M3_CORROB_TOL_M:
                return "ACCEPT", (reason + f" + cadastre-corroborated {cadastre_corrob_m:.1f}m "
                                  f"(2 independent sources agree)")
            return "REVIEW", (reason + f" BUT disagrees with the cadastre seat by "
                              f"{cadastre_corrob_m:.0f}m (>{M3_CORROB_TOL_M:.0f}m) -> confirm")
        return "ACCEPT", reason

    if n_matched >= review_bar and median_resid <= M3_REVIEW_RESIDUAL_MEDIAN_M:
        reason = "located; confirm placement" if scale_ok else "located; scale off -> confirm"
        if not tiles:
            reason = "located but overlaps a placed plot -> confirm extent"
        return "REVIEW", f"n={n_matched}/{n_corners} med={median_resid:.2f}m ({reason})"

    return "NEEDS_GPS", (f"n={n_matched}/{n_corners} med={median_resid:.2f}m "
                         f"(stones present but fit too weak -> needs GPS/operator seed)")


# ----------------------------------------------------------------- deliverables ----
_LAYER_ACI = {"ACCEPT": 3, "ACCEPT_SEEDED": 5, "ACCEPT_RELATIVE": 4, "REVIEW": 30,
              "NEEDS_GPS": 1}                                        # grn/blue/cyan/org/red
_DISP_COLOR = {"ACCEPT": "#2ca02c", "ACCEPT_SEEDED": "#1f77b4", "ACCEPT_RELATIVE": "#17becf",
               "REVIEW": "#ff7f0e", "NEEDS_GPS": "#d62728", "UNMEASURED": "#7f7f7f"}


def write_dxf(placements: list[M3Placement], out_path: Path, *,
              crs: str = "EPSG:32643") -> Path:
    """One georeferenced DXF: every PLACED plot's outer ring, layered by disposition.
    UNMEASURED plots have no placement and are skipped."""
    import ezdxf
    out_path = Path(out_path)
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    for p in placements:
        if p.ring_utm is None or len(p.ring_utm) < 3:
            continue
        pts = [(float(x), float(y)) for x, y in p.ring_utm]
        pts.append(pts[0])
        msp.add_lwpolyline(pts, dxfattribs={
            "layer": f"M3_{p.disposition}",
            "color": _LAYER_ACI.get(p.disposition, 7)})
        cx, cy = float(np.mean(p.ring_utm[:, 0])), float(np.mean(p.ring_utm[:, 1]))
        label = f"{p.village}:{p.survey_number}" if p.village else str(p.survey_number)
        msp.add_text(label,
                     dxfattribs={"height": 3.0, "layer": f"M3_{p.disposition}"}
                     ).set_placement((cx, cy))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(out_path))
    return out_path


def write_overlay(placements: list[M3Placement], stones_xy: np.ndarray,
                  out_path: Path, *, village: str = "") -> Path:
    """One QA PNG: the surveyor stone cloud (grey dots) with each placed plot coloured
    by disposition. An operator eyeballs whether the ACCEPT plots sit on the stones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(16, 12))
    if stones_xy is not None and len(stones_xy):
        ax.scatter(stones_xy[:, 0], stones_xy[:, 1], s=1.0, c="#bbbbbb",
                   marker=".", label="surveyor stones", zorder=1)
    patches, colors = [], []
    for p in placements:
        if p.ring_utm is None or len(p.ring_utm) < 3:
            continue
        patches.append(MplPolygon(p.ring_utm, closed=True))
        colors.append(_DISP_COLOR.get(p.disposition, "#000000"))
    if patches:
        ax.add_collection(PatchCollection(patches, facecolors=colors, edgecolors="black",
                                          linewidths=0.4, alpha=0.55, zorder=2))
    n_acc = sum(1 for p in placements if p.disposition == "ACCEPT")
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(f"M3 QA overlay {village} -- {n_acc}/{len(placements)} ACCEPT "
                 f"(green=ACCEPT orange=REVIEW red=NEEDS_GPS)")
    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_report(placements: list[M3Placement], out_path: Path, *,
                 village: str = "") -> Path:
    """Per-plot JSON audit: disposition + the evidence the gate saw (matched count,
    median/max residual, diagnostic scale). Honest counts, nothing inflated."""
    out_path = Path(out_path)
    rows = []
    counts: dict[str, int] = {}
    for p in placements:
        counts[p.disposition] = counts.get(p.disposition, 0) + 1
        rows.append({
            "village": p.village,
            "survey_number": p.survey_number,
            "disposition": p.disposition,
            "n_matched": p.n_matched,
            "n_corners": p.n_corners,
            "median_residual_m": (None if p.median_residual_m != p.median_residual_m
                                  else round(p.median_residual_m, 3)),
            "max_residual_m": (None if p.max_residual_m != p.max_residual_m
                               else round(p.max_residual_m, 3)),
            "s_fitted": (None if p.s_fitted != p.s_fitted else round(p.s_fitted, 4)),
            "scale_locked_to": 1.0,
            "cadastre_corrob_m": (None if p.cadastre_corrob_m != p.cadastre_corrob_m
                                  else round(p.cadastre_corrob_m, 2)),
            "note": p.note,
        })
    rows.sort(key=lambda r: (r["disposition"] != "ACCEPT",
                             int(r["survey_number"]) if str(r["survey_number"]).isdigit()
                             else 1_000_000_000, str(r["survey_number"])))
    n_acc = counts.get("ACCEPT", 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "village": village,
        "summary": (f"{n_acc}/{len(placements)} ACCEPT (survey-grade, scale-locked); "
                    f"dispositions {counts}"),
        "disposition_counts": counts,
        "plots": rows,
    }, indent=2))
    return out_path
