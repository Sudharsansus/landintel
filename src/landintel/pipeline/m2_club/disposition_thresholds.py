"""Single source of truth for the M2-club + agent-layer disposition thresholds.

One edit here moves the precision/recall point of the whole M2 stage. Every constant
is GENERAL -- keyed on geometry/data, never on a village -- and each carries the
rationale for its value so a future tuning session argues with the reason, not the
number.

Scope note: the M3 surveyor-corridor gates (chain_coverage 0.50/0.35, inlier counts,
residual bounds) intentionally stay in ``m2_georef.pipeline`` -- that path is
INGUR-validated and frozen; centralizing M2 must not touch it.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------- tiling overlap
# Interior overlap (fraction of the SMALLER footprint) above which two confident
# plots cannot both stand. VALIDATED at 0.30: FMB-preserved neighbours legitimately
# share edges, and on a thin sliver parcel the shared-edge intersection can reach
# ~20-25% of the smaller footprint (measured: a correct thin parcel was falsely
# demoted at 0.20 for a 22% shared-edge overlap). 0.30 separates "shared edge"
# from "stacked on the same land" across both datasets. Previously this value
# DRIFTED across four call sites (0.20 in two, 0.30 in two) -- a silent FP-policy
# inconsistency; every consumer now imports THIS constant.
TILING_OVERLAP_THRESHOLD: float = 0.30

# ------------------------------------------------------- cadastral rigid shape gate
# (identical bands to the validated M3 off-corridor cadastral path)
CAD_AREA_LO: float = 0.65    # placed M1 area / parcel area lower bound (right-sized)
CAD_AREA_HI: float = 1.55    # upper bound
CAD_SCALE_LO: float = 0.80   # leftover rigid scale (M1 metres -> parcel metres ~ 1)
CAD_SCALE_HI: float = 1.25
CAD_ROT_RESID_MAX: float = 12.0  # rigid corner-alignment residual FLOOR (m)
CAD_ROT_K: float = 0.30      # residual tolerance per metre of placed equivalent
                             # radius (raster corner jitter scales with parcel size)

# ------------------------------------------------------------- seat-locality lock
# A shape-gate-passing fit must also seat the plot back on its own label point,
# else it is a wrong-but-same-size parcel collision. Tolerance is PARCEL-SIZE-
# RELATIVE (SEAT_K equivalent-radii), floored for OCR label-point jitter.
SEAT_K: float = 1.6
SEAT_FLOOR_M: float = 60.0

# ------------------------------------------------------------ stone-count confidence
# A rigid pose (rotation + translation, scale locked 1) is only well-CONSTRAINED by
# >= this many corner correspondences; fewer is minimal/under-determined and one
# jittered stone tilts the whole placement. Under club-all M2 this is a CONFIDENCE
# bar (high vs low confidence label), never a placement blocker -- a plot with fewer
# corners is still placed best-effort. The bar is DATA-KEYED: a plot can never be
# asked for more stones than it has (callers use min(bar, n_corners)).
CAD_MIN_STONES: int = int(os.environ.get("LANDINTEL_CAD_MIN_STONES", "4"))

# Full-confidence stone-match bar (the client's "make it 5"): a placement matching
# at least min(FULL_MATCH_STONES, n_corners) target stones earns the top confidence
# tier. Conditional on the plot's own corner count, so a 4-corner plot is judged by
# all 4 of its corners -- never silently excluded by a flat constant.
FULL_MATCH_STONES: int = int(os.environ.get("LANDINTEL_FULL_MATCH_STONES", "5"))

# ------------------------------------------------ stone-coincidence tolerance
# How close a placed FMB corner must land to a cadastre parcel corner to count as
# "the same stone". SIZE-RELATIVE: the FMB-vs-cadastre SHAPE disagreement grows
# with parcel size (a few percent of extent), so the tolerance is a fraction of
# the parcel's equivalent radius with an absolute floor for small parcels /
# vector-trace noise. Data-keyed, never a per-village number.
STONE_TOL_FLOOR_M: float = 3.0
STONE_TOL_K: float = 0.08     # per metre of parcel equivalent radius
