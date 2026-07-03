"""M2 method 1 -- seat one FMB on its authoritative cadastral parcel.

Surveyor-free. Identity is keyed by SURVEY NUMBER (no shape search): the cadastral
source (TNGIS tiles or a client vector file) maps survey# -> official UTM parcel,
and we rigidly place the M1 FMB onto it (rotation + uniform scale ~1 + translation,
geometry preserved, NEVER warped onto the raster). The strict RIGID SHAPE GATE
(area ratio in band + scale ~1 + orientation not flipped + small corner residual)
is the 0-FP arbiter: a wrong-parcel or wrong-size identity collision fails it.

This is the same rigid-fit machinery the M3 surveyor pipeline already uses for
off-corridor cadastral placement (``m5_cadastral.fit.fit_plot_to_parcel``), lifted
OUT of the surveyor-coupled path so it can stand alone as an M2 coordinate source.
"""
from __future__ import annotations

import logging
import os

import numpy as np

from ..m2_georef.extract_m1 import M1PlotData
from ..m5_cadastral.fit import fit_plot_to_parcel
from .placement import CandidatePlacement

_log = logging.getLogger(__name__)

# Robust corner residual (adopted from the M2 open-source "diamonds" review -- a
# partial-Chamfer / Modified-Hausdorff trim of the plain mean residual). OFF by default:
# the plain residual is the shipped, INGUR-validated behaviour, and per the note above
# rot_residual is already redundant for FP rejection (seat-locality is the FP lock), so the
# robust variant can only RECOVER recall on jittered-corner fits, never add a false accept.
# Flip on (LANDINTEL_CAD_ROBUST_RESID=1) once an INGUR regression confirms +recall / 0-FP.
CAD_ROBUST_RESID = os.environ.get("LANDINTEL_CAD_ROBUST_RESID", "0") == "1"


def _gate_residual(fit) -> float:
    """The residual the shape gate compares against tolerance. With the robust switch on,
    a jittered corner cannot inflate it past a genuinely-good fit's value: we take the
    lower of the plain mean and the trimmed-mean residual, so it is strictly recall-additive
    (a fit already passing on the mean is unchanged; one failing ONLY on a noisy corner can
    now pass). The FP lock (area/scale/orientation + seat-locality) is untouched either way."""
    r = float(fit.rot_residual)
    if CAD_ROBUST_RESID:
        rr = float(getattr(fit, "rot_residual_robust", float("inf")))
        if rr == rr:  # not NaN
            return min(r, rr)
    return r

# Rigid shape gate (identical bands to the validated M3 cadastral path). A placement
# is gate-passing ONLY if all hold; otherwise the candidate is still returned but
# flagged passes_gate=False, so the pipeline surfaces it as REVIEW, never ACCEPT.
CAD_AREA_LO = 0.65          # placed M1 area / parcel area lower bound (right-sized)
CAD_AREA_HI = 1.55          # upper bound
CAD_SCALE_LO = 0.80         # leftover rigid scale (M1 is metres, parcel is metres -> ~1)
CAD_SCALE_HI = 1.25
# PLACE AT TRUE SCALE. The fitted scale (~0.95) is the raster cadastre being slightly off, NOT
# the FMB -- M1 is already real metres, so the survey's true scale to UTM is 1. Baking the
# fitted scale into the geometry SHRINKS every edge 2-5% (measured: perimeters 0.93-0.99 of
# M1), corrupting the survey lengths. So the scale is used ONLY to GATE (does the plot belong
# to this parcel?) and the geometry is emitted as a PURE RIGID body (scale=1, edge lengths
# exactly preserved), anchored at the fitted centroid. This is the client's "one base point +
# club the boundary, keep the lengths" directive.
CAD_RIGID_SCALE = True
CAD_ROT_RESID_MAX = 12.0    # rigid corner-alignment residual FLOOR (m); gross misfit -> REVIEW
# The rigid corner-alignment residual is measured against a RASTER-traced parcel polygon
# (z18 ~0.586 m/px, coarse approxPolyDP corners), so its absolute value scales with parcel
# size: the same fractional corner jitter is ~3x larger on a 120 m parcel than a 40 m one.
# A flat 12 m floor under-accepts large-but-correct placements (measured: surveys 668/1024/
# 1025 fail ONLY rot_residual at 14-24 m while area/scale/orientation AND the seat-locality
# lock all pass strongly). So the tolerance is FLOORED at CAD_ROT_RESID_MAX and grows with the
# placed parcel's equivalent radius. This does NOT weaken 0-FP: seat-locality (centroid on its
# own label) + area band + scale~1 + orientation are the false-positive lock; rot_residual was
# always redundant for FP rejection (it never stopped the 27 cross-placements -- seat did).
CAD_ROT_K = 0.30            # rot-residual tolerance per metre of placed equivalent radius
                            # (within the principled 0.25-0.35 raster-coarseness band; seat-
                            # locality + area + scale + orientation remain the FP lock)
# SEAT-LOCALITY GATE -- the decisive 0-FP lock. The rigid shape gate ALONE over-matches in a
# field of similar-sized blobby parcels (measured: 27/156 wrong cross-placements pass it). The
# discriminator is that a CORRECT placement seats the M1 plot back ON its own OCR label point.
# The tolerance is PARCEL-SIZE-RELATIVE (not an absolute metre value, which is too loose for a
# small urban parcel and too tight for a large rural one): a placement may sit within SEAT_K
# equivalent-radii of the label (radius = sqrt(area/pi)), floored at SEAT_FLOOR_M for OCR
# label-point jitter. It only ADDS a rejection; it never promotes. (Resolution/zone-agnostic.)
SEAT_K = 1.6
SEAT_FLOOR_M = 60.0
# MINIMUM CORNER-STONE MATCH for a confident cadastral ACCEPT. A rigid pose (rotation +
# translation, scale~1) is only well-CONSTRAINED by >= 4 corner correspondences; a 3-stone fit
# is minimal and a single jittered stone tilts the whole placement (the visible "gap"). So a plot
# fit with fewer than this is still PLACED (located) but routed to REVIEW, never ACCEPT. All
# current village plots already carry >= 4 corners, so this tightens the standard without loss.
# Override via env (LANDINTEL_CAD_MIN_STONES); default 4.
CAD_MIN_STONES = int(os.environ.get("LANDINTEL_CAD_MIN_STONES", "4"))


def _placed_area(fit, m1) -> float:
    """Area (m^2) of the rigidly-placed M1 corner ring -- the basis for size-relative gates."""
    from shapely.geometry import Polygon
    ring = fit.adjusted[np.array(m1.outer_stone_indices)]
    p = Polygon([(float(x), float(y)) for x, y in ring])
    if not p.is_valid:
        p = p.buffer(0)
    return float(p.area) if hasattr(p, "area") else 0.0


def _rot_resid_tol(placed_area: float) -> float:
    """Size-relative rigid-residual tolerance: floored at CAD_ROT_RESID_MAX, grows with the
    placed parcel's equivalent radius (raster corner jitter scales with parcel size)."""
    eq_radius = float(np.sqrt(max(placed_area, 1.0) / np.pi))
    return max(CAD_ROT_RESID_MAX, CAD_ROT_K * eq_radius)


def _rigidify(fit, m1):
    """Re-derive the placement as a PURE rigid transform (scale locked to 1) so the FMB's
    true survey edge lengths are preserved EXACTLY, keeping the plot centred where the scaled
    fit placed it. Returns (R, t, adjusted). ``fit`` maps orig -> adjusted as s*R*orig + t;
    the rigid version is R*orig + t' with t' = t + (s-1)*R*c (c = orig centroid), so the
    centroid is unchanged but every edge length equals the original M1 length."""
    R = np.asarray(fit.R, float)
    s = float(fit.s)
    t = np.asarray(fit.t, float)
    orig = np.asarray(m1.stone_positions(), float)
    c = orig.mean(axis=0)
    t_rigid = t + (s - 1.0) * (R @ c)
    adjusted = (R @ orig.T).T + t_rigid
    return R, t_rigid, adjusted


def _passes_shape_gate(fit, m1) -> bool:
    if fit is None or fit.method != "rigid":
        return False
    # Require the rigid fit to be constrained by >= CAD_MIN_STONES corner stones. Fewer than
    # that is an under-determined pose -> located but REVIEW, never a confident ACCEPT.
    if len(m1.outer_stone_indices) < CAD_MIN_STONES:
        return False
    if not (CAD_AREA_LO <= fit.area_ratio <= CAD_AREA_HI
            and CAD_SCALE_LO <= fit.s <= CAD_SCALE_HI
            and fit.orientation_ok):
        return False
    return _gate_residual(fit) <= _rot_resid_tol(_placed_area(fit, m1))


def cadastral_seat(
    m1: M1PlotData,
    cadastral_source,
) -> CandidatePlacement | None:
    """Place ``m1`` on its cadastral parcel by survey number, rigidly.

    Returns a ``CandidatePlacement`` (``method="cadastral"``) or None when the
    source has no parcel for this survey / the fit is unusable. ``passes_gate``
    reflects the strict rigid shape gate -- the pipeline upgrades to ACCEPT only
    when it is True (or another independent method corroborates the position).

    When the primary parcel ring fails the gate, any locally-reconstructed
    ``recovered_candidates`` rings are tried and the best gate-passing one adopted
    (the gate is the sole arbiter, so a recovered ring can never force a false
    ACCEPT). This mirrors the M3 cadastral recovery path.
    """
    if cadastral_source is None or not m1.survey_number:
        return None
    parcel = cadastral_source.get(m1.survey_number)
    if parcel is None:
        return None
    if len(m1.outer_stone_indices) < 3:
        return None

    label_point = getattr(cadastral_source, "label_point", lambda s: None)
    anchor = label_point(m1.survey_number)

    fit = fit_plot_to_parcel(m1, parcel, anchor=anchor)
    if fit is None:
        return None

    if not _passes_shape_gate(fit, m1):
        # Open-parcel recovery: try alternative reconstructed rings, keep the best
        # gate-passing one (closest area ratio to 1). Adds recall, never an FP.
        get_cands = getattr(cadastral_source, "recovered_candidates", lambda s: [])
        best = None
        for cand in get_cands(m1.survey_number):
            cf = fit_plot_to_parcel(m1, cand, anchor=anchor)
            if _passes_shape_gate(cf, m1):
                if best is None or abs(cf.area_ratio - 1.0) < abs(best.area_ratio - 1.0):
                    best = cf
        if best is not None:
            fit = best

    passes = _passes_shape_gate(fit, m1)   # gate on the SCALED fit (validates parcel membership)
    note = ""

    # Emit a PURE RIGID placement (scale=1) so survey edge lengths are preserved exactly;
    # the fitted scale only gated above. Everything downstream (seat-locality, output) uses it.
    if CAD_RIGID_SCALE:
        R_p, t_p, adj_p = _rigidify(fit, m1)
        s_p = 1.0
    else:
        R_p, t_p, adj_p, s_p = fit.R, fit.t, fit.adjusted, float(fit.s)

    # Seat-locality gate (0-FP lock): a shape-gate-passing fit must ALSO seat the plot back
    # on its own label point, else it is a wrong-but-same-size parcel collision -> REVIEW.
    # Tolerance scales with the placed parcel's size (small parcels -> tight, large -> generous).
    if passes and anchor is not None and len(m1.outer_stone_indices) >= 3:
        from shapely.geometry import Polygon
        ring = adj_p[np.array(m1.outer_stone_indices)]
        centroid = ring.mean(axis=0)
        placed_area = Polygon([(float(x), float(y)) for x, y in ring]).area
        seat_tol = max(SEAT_FLOOR_M, SEAT_K * float(np.sqrt(max(placed_area, 1.0) / np.pi)))
        seat_dist = float(np.hypot(centroid[0] - anchor[0], centroid[1] - anchor[1]))
        if seat_dist > seat_tol:
            passes = False
            note = (f"off-seat: placed {seat_dist:.0f} m from its label point "
                    f"(> {seat_tol:.0f} m, scaled to parcel) -> wrong-parcel collision")

    # An aggressive-recovered parcel (wider 2nd pass, leaking net) is less reliable -> the plot
    # is LOCATED but routed to REVIEW, never ACCEPT (strictly additive recall, 0-FP).
    if passes and getattr(cadastral_source, "is_aggressive", lambda s: False)(m1.survey_number):
        passes = False
        note = "aggressive-recovered parcel (leaking boundary net) -> located, REVIEW not ACCEPT"

    if not passes and not note:
        reasons = []
        if fit.method != "rigid":
            reasons.append(f"method={fit.method}")
        if not (CAD_AREA_LO <= fit.area_ratio <= CAD_AREA_HI):
            reasons.append(f"area_ratio={fit.area_ratio:.2f}")
        if not (CAD_SCALE_LO <= fit.s <= CAD_SCALE_HI):
            reasons.append(f"scale={fit.s:.3f}")
        if not fit.orientation_ok:
            reasons.append("flipped")
        if _gate_residual(fit) > _rot_resid_tol(_placed_area(fit, m1)):
            rr = float(getattr(fit, "rot_residual_robust", float("nan")))
            robust_note = f" (robust {rr:.1f}m)" if rr == rr and rr < fit.rot_residual else ""
            reasons.append(f"rot_resid={fit.rot_residual:.1f}m{robust_note}")
        note = "below cadastral shape gate: " + ", ".join(reasons)

    return CandidatePlacement(
        method="cadastral",
        R=R_p, s=s_p, t=t_p,
        adjusted=adj_p,
        corner_ring=list(m1.outer_stone_indices),
        passes_gate=passes,
        area_ratio=fit.area_ratio,
        rot_residual=fit.rot_residual,
        scale=float(fit.s),          # report the FITTED scale (diagnostic); placement is rigid
        note=note,
    )
