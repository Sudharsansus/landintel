"""M3 -- georeference M1 FMBs against the SURVEYOR raw-data stones (survey-grade).

PARALLEL to M2, never downstream of it. Inputs: M1 FMB DXFs + surveyor DXF (+ Google
for the village LOCATION only). NO M2 output, NO cadastre matching -- so an M2 error
can never cascade into M3, and M3 can place the very plots M2 gets wrong.

Two phases, both cadastre/M2-free:
  1. ANCHOR  -- each M1 plot's corner shape is matched geometrically to the cropped
     surveyor stones; plots that match strongly + uniquely self-locate (a distinctive
     corner N-gon needs no seed).
  2. GROW    -- unplaced plots are matched against surveyor stones NEAR the already-
     placed plots (seed = M3's OWN anchored plots, not M2), and kept only if the new
     placement tiles (does not overlap a placed plot). Iterates outward until stable.

Final coordinates come 100% from the surveyor stones -> survey-grade.

Usage:  python run_m3.py <VILLAGE> --surveyor "<RAW DATA.dxf>" [--district Erode --taluk ..]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")


def _load_dotenv(path: str = ".env") -> None:
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_dotenv()

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon

from landintel.pipeline.m2_georef.extract_m1 import extract_m1_dxf
from landintel.pipeline.m2_georef.extract_surveyor import extract_surveyor
from landintel.pipeline.m2_georef.match import geometric_match
from landintel.pipeline.m2_georef.m3_deliverables import (
    M3Placement, M3_CORROB_TOL_M, M3_ACCEPT_RESIDUAL_MEDIAN_M, M3_ACCEPT_RESIDUAL_MAX_M,
    classify, place_scale_locked, write_dxf, write_overlay, write_report)
from landintel.pipeline.m2_club.disposition_thresholds import CAD_MIN_STONES, FULL_MATCH_STONES
from landintel.pipeline.m5_cadastral.geo_locate import _google_geocode_candidates

CRS = "EPSG:32643"
# Confident self-locating ANCHOR bar (data-keyed, general): a distinctive corner subset
# that cannot chance-match -- >= ANCHOR_MIN_INLIERS stones AND >= ANCHOR_MIN_FRAC of the
# plot's own corners at sub-ANCHOR_MAX_RESID residual.
ANCHOR_MIN_INLIERS = 6
ANCHOR_MIN_FRAC = 0.45
ANCHOR_MAX_RESID = 3.0
# GROW acceptance: a neighbour placed by locality needs a real match too, but the tiling
# (non-overlap) constraint is the extra evidence, so the inlier bar is min(4, n_corners).
GROW_MIN_INLIERS = 4
GROW_MAX_RESID = 3.5
GROW_OVERLAP_MAX = 0.30          # placed plots tile; >30% interior overlap = wrong seat
GROW_REGION_PAD = 60.0           # metres beyond a placed plot's radius to admit neighbours
# Cadastre-SEEDED crop margin: metres added beyond a plot's own radius to size the local
# surveyor-stone window around its cadastre seat. Sized to the cadastre seat's uncertainty
# (~15-20 m: raster/vector registration + the FMB-vs-field 3-5 m discrepancy), while the
# chance-congruent decoys the whole-village search hits are ~700 m away -- so this window sits
# in a ~100x gap and captures the RIGHT local stones without a per-village constant.
SEED_CROP_MARGIN = 40.0


def _cadastre_seeds(village, anchor_latlon, ax, ay, surveys, m1_paths, crs):
    """Per-survey cadastre seat = where the FMB SITS on its own parcel, a LOCATION PRIOR only
    (for cropping + a corroboration cross-check). M3 computes it ITSELF by running the shared M2
    CLUB ALGORITHM (`club_pipeline`) on the M1 DXFs + an independently-built TNGIS vector source
    -- this reuses M2's ALGORITHM, never M2's OUTPUT files, so an M2 run's mistakes can never
    cascade here (a single-parcel `cadastral_seat` fit alone mis-places large plots by >200 m;
    the full club's fit+snap+propagation is what lands them correctly). Fail-safe: returns {}
    when the parquet/anchor is unavailable, so M3 falls back to blind anchor+grow with no
    regression. The seat only CROPS + CROSS-CHECKS; the surveyor stones set every coordinate
    (rule 2, cadastre-crop-only)."""
    import shutil
    import tempfile
    try:
        parq = os.environ.get("LANDINTEL_TNGIS_PARQUET",
                              "data/tngis/TNGIS_TN_Cadastrals.parquet")
        if not Path(parq).exists():
            return {}
        from landintel.pipeline.m2_club import club_pipeline
        from landintel.pipeline.m5_cadastral.vector_locate import (
            load_area_parcels_cached, village_candidates)
        ck = f"data/tngis/area_{village}_{anchor_latlon[0]:.2f}_{anchor_latlon[1]:.2f}.json"
        parcels = load_area_parcels_cached(anchor_latlon, cache_json=ck)
        cands = village_candidates(parcels, set(surveys), (ax, ay), radius_m=5000.0,
                                   min_overlap=3, max_cand=6, crs=crs)
        if not cands:
            return {}
        # Top candidate = most FMB-survey overlap, then nearest to the anchor. No IoU pick is
        # needed here (unlike M2): the surveyor stones + corroboration are the real gate, so a
        # wrong-village seed only yields empty crops / no corroboration -> falls back, 0-FP.
        src = cands[0]["source"]
        tmp = Path(tempfile.mkdtemp(prefix="m3seed_"))         # discarded; M2 output untouched
        try:
            results = club_pipeline(m1_paths, tmp, crs=crs, cadastral_source=src, village=village)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        seeds = {}
        for r in results:
            fp = r.placement.footprint() if r.placement is not None else None
            if fp is None or fp.is_empty:
                continue
            c = fp.centroid
            seeds[str(r.survey_number)] = (float(c.x), float(c.y))
        return seeds
    except Exception as exc:  # noqa: BLE001 - seeding is best-effort; never break M3
        print(f"[seed] cadastre seeding unavailable ({exc}); blind anchor+grow only")
        return {}


def _place(m1, surveyor, result):
    """SCALE-LOCKED rigid placement (rule 2) from a MatchResult's stone_map.

    Builds the matched (M1 corner -> surveyor stone) pairs and fits ROTATION + TRANSLATION
    only (scale == 1, via rigid_procrustes) -- FMB edge lengths are preserved exactly. The
    similarity scale umeyama WOULD have used is kept as a DIAGNOSTIC (``s_fitted``) and is
    never applied; a value far from 1 flags an upstream M1 unit bug. Returns a placement
    dict or None (fewer than 2 pairs, or the diagnostic scale is insane -> not a real match).
    """
    pos = m1.stone_positions()
    surv = surveyor.stone_positions
    src, dst = [], []
    for i, j in enumerate(result.stone_map):
        if j is not None and j >= 0:
            src.append(pos[i]); dst.append(surv[j])
    if len(src) < 2:
        return None
    R, t, s_fitted, residuals = place_scale_locked(np.array(src, float), np.array(dst, float))
    if not (0.5 < s_fitted < 2.0):        # diagnostic sanity: a true rigid match has s ~ 1
        return None
    ring = (pos @ R.T + t)[np.array(m1.outer_stone_indices)]   # scale-1 (rigid) placement
    return {"R": R, "t": t, "s_fitted": s_fitted, "ring": ring,
            "residuals": residuals, "n_matched": len(src)}


def _place_relative(m1_b, cand, anchor_corners, tol=3.0):
    """RULE-2 rigid (scale=1) placement of a plot propagated onto a confirmed neighbour.

    ``propagate_geometric`` (the client's FMBS_STONES_MATCH) finds WHICH of ``m1_b``'s FMB
    corners coincide with the anchor's placed corners, but it applies a small umeyama scale
    (0.85-1.18). M3 is scale-strict, so we take those shared correspondences and re-fit
    ROTATION + TRANSLATION only (``place_scale_locked``): the FMB edge lengths are preserved
    exactly and the plot inherits the anchor's georeference at the shared boundary. The fitted
    scale is a diagnostic; far from 1 would be an upstream unit bug, never baked in. Returns a
    placement dict like ``_place`` (with ``relative_to`` set by the caller), or None."""
    pos = m1_b.stone_positions()
    b_adj = np.asarray(cand.adjusted, float)
    src, dst, used = [], [], set()
    for bi in m1_b.outer_stone_indices:
        if not (0 <= bi < len(b_adj)):
            continue
        d = np.hypot(anchor_corners[:, 0] - b_adj[bi][0], anchor_corners[:, 1] - b_adj[bi][1])
        ai = int(np.argmin(d))
        if d[ai] <= tol and ai not in used:                # same physical corner
            src.append(pos[bi]); dst.append(anchor_corners[ai]); used.add(ai)
    if len(src) < 3:                                       # client's >=3 shared-stone bar
        return None
    R, t, s_fitted, residuals = place_scale_locked(np.array(src, float), np.array(dst, float))
    if not (0.5 < s_fitted < 2.0):
        return None
    ring = (pos @ R.T + t)[np.array(m1_b.outer_stone_indices)]
    return {"R": R, "t": t, "s_fitted": s_fitted, "ring": ring,
            "residuals": residuals, "n_matched": len(src), "relative_to": None}


def _fp(ring):
    if ring is None or len(ring) < 3:
        return None
    p = Polygon([(float(x), float(y)) for x, y in ring])
    if not p.is_valid:
        p = p.buffer(0)
    return p if (not p.is_empty and p.area > 0) else None


def _overlaps(fp, placed):
    for pf in placed.values():
        if pf is None or not fp.intersects(pf):
            continue
        if fp.intersection(pf).area / max(min(fp.area, pf.area), 1e-9) > GROW_OVERLAP_MAX:
            return True
    return False


def place_village(village, surveyor, district="Erode", taluk=""):
    """Place ALL of ONE village's M1 FMBs onto the surveyor stone cloud by sequential verified
    growth (the client's 'match ONE FMB, verify, commit, then move to the next'). Returns
    ``(list[M3Placement], surveyor stone positions in the village window)``. A reusable METHOD:
    both the per-village run and the SINGLE combined-district output call it and the caller writes
    the deliverables -- so all villages can be clubbed onto the shared stone cloud in one DXF."""
    g = _google_geocode_candidates(f"{village}, {taluk}, {district}, Tamil Nadu, India", 1)
    if not g:
        raise SystemExit(f"Google geocode failed for {village}")
    ax, ay = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(g[0][1], g[0][0])
    print(f"[1/4] Google anchor {village} @ ({g[0][0]:.5f},{g[0][1]:.5f})")

    m1s = {}
    m1_paths = sorted(str(p) for p in Path(f"output/{village}/m1").glob("*.dxf"))
    for p in m1_paths:
        m1 = extract_m1_dxf(p)
        if len(m1.outer_stone_indices) >= 3:
            m1s[str(m1.survey_number)] = m1

    # --- Cadastre seeds FIRST (surveyor-independent): per-survey parcel positions from the
    # shared M2 club algorithm on an independently-built TNGIS source (never M2's output). ---
    seeds = _cadastre_seeds(village, (g[0][0], g[0][1]), ax, ay, set(m1s), m1_paths, CRS)

    # Surveyor window: PIN it on the seeds' own extent when available (+ pad for plot radius),
    # not the coarse village-name geocode. The village-name point can be ~2 km off the actual
    # parcels (measured: NASIYANUR's Google pin is 1.8 km from its cadastre block, outside a
    # 1500 m anchor window) -- the cadastre gives the true per-survey positions, so a seed-pinned
    # window reliably covers the real stones. Falls back to the anchor window with no seeds.
    if seeds:
        sx = [p[0] for p in seeds.values()]; sy = [p[1] for p in seeds.values()]
        pad = 700.0
        bbox = (min(sx) - pad, min(sy) - pad, max(sx) + pad, max(sy) + pad)
    else:
        pad = 1500.0
        bbox = (ax - pad, ay - pad, ax + pad, ay + pad)
    sd = extract_surveyor(surveyor, bbox=bbox); sd.build_index()
    spos = sd.stone_positions
    print(f"[2/4] {len(sd.stones)} surveyor stones in the {village} window "
          f"({'seed-pinned on the cadastre block' if seeds else 'anchor-pinned'})")

    from landintel.pipeline.m2_club.placement import CandidatePlacement
    from landintel.pipeline.m2_club.relative_club import propagate_geometric

    placed, placed_fp = {}, {}

    # ONE plot's best DIRECT stone placement: cadastre-seeded crop (de-saturated) plus a
    # whole-window self-locate, keeping whichever has the smaller median residual. RANSAC is
    # expensive, so this is computed ONCE per plot and cached; the sequential loop below only
    # re-checks the cheap tiling constraint each round.
    def _direct_place(sv, m1):
        seat = seeds.get(sv)
        cands = []
        if seat is not None:
            ring0 = m1.stone_positions()[np.array(m1.outer_stone_indices)]
            prad = float(np.hypot(np.ptp(ring0[:, 0]), np.ptp(ring0[:, 1]))) * 0.6
            d = np.hypot(spos[:, 0] - seat[0], spos[:, 1] - seat[1])
            allowed = d <= (prad + SEED_CROP_MARGIN)
            if allowed.sum() >= 4:
                gm = geometric_match(m1, sd, allowed_stones=allowed)
                if gm.matched:
                    p = _place(m1, sd, gm)
                    if p is not None:
                        cands.append(p)
        gm = geometric_match(m1, sd)                       # whole-window self-locate
        if gm.matched:
            p = _place(m1, sd, gm)
            if p is not None:
                cands.append(p)
        best = None
        for p in cands:
            if seat is not None:
                pc = p["ring"].mean(axis=0)
                p["corrob"] = float(np.hypot(pc[0] - seat[0], pc[1] - seat[1]))
            med = float(np.median(p["residuals"])) if len(p["residuals"]) else float("inf")
            if best is None or med < best[1]:
                best = (p, med)
        return best[0] if best else None

    direct = {sv: _direct_place(sv, m1) for sv, m1 in m1s.items()}

    def _verified(sv, pl):
        """Multi-source VERIFICATION of ONE placement -- the client's 'match + verify before the
        next'. A plot is committed only when independent sources AGREE on it; accuracy (the robust
        MEDIAN residual) is never relaxed:
          - survey-grade median residual  AND  well-constrained pose (>= min(CAD_MIN_STONES,n)), AND
          - EITHER the placement agrees with the plot's OWN government cadastre seat (survey# ->
            parcel, <= M3_CORROB_TOL_M -- an independent 2nd source),
            OR a dense clean self-match (>= min(FULL_MATCH_STONES,n) stones AND worst corner <= 3 m).
        Tiling vs the committed set is checked by the caller. Returns (ok, score)."""
        n = len(m1s[sv].outer_stone_indices)
        res = pl.get("residuals", [])
        if len(res) == 0:
            return False, 0.0
        med = float(np.median(res)); mx = float(np.max(res))
        # SURVEY-GRADE at EVERY corner (median AND worst corner within the validated bounds) on a
        # dense pose -- the accuracy bar is NEVER relaxed. A single corner beyond 3 m is FMB-vs-
        # field noise: the plot is still correctly LOCATED (handled as a lower tier), but it is not
        # survey-grade, so it does not earn ACCEPT.
        if not (pl["n_matched"] >= min(FULL_MATCH_STONES, n)
                and med <= M3_ACCEPT_RESIDUAL_MEDIAN_M
                and mx <= M3_ACCEPT_RESIDUAL_MAX_M):
            return False, 0.0
        # AND independently cadastre-confirmed (2nd source). A plot with a seat MUST agree with it
        # -- a dense self-match that disagrees is a saturated-field DECOY, not a confirmation.
        corrob = pl.get("corrob")
        if corrob is None or corrob > M3_CORROB_TOL_M:
            return False, 0.0
        score = pl["n_matched"] - med - 0.02 * corrob
        return True, score

    # --- SEQUENTIAL VERIFIED GROWTH (client's method: match ONE FMB, VERIFY, commit, next) ---
    # No batch "place all then judge". Each round commits exactly ONE more plot -- the single
    # best INDEPENDENTLY-VERIFIED candidate across all unplaced FMBs -- then re-evaluates tiling
    # with it committed. Nothing is trusted until multiple sources agree on it.
    while True:
        best = None                                        # (sv, pl, fp, score)
        for sv, m1 in m1s.items():
            if sv in placed:
                continue
            pl = direct.get(sv)
            if pl is None:
                continue
            fp = _fp(pl["ring"])
            if fp is None or _overlaps(fp, placed_fp):
                continue
            ok, score = _verified(sv, pl)
            if ok and (best is None or score > best[3]):
                best = (sv, pl, fp, score)
        if best is None:
            break
        sv, pl, fp, _s = best
        placed[sv], placed_fp[sv] = pl, fp                 # commit ONE verified plot, then re-loop
    print(f"[3/4] verified growth: {len(placed)}/{len(m1s)} plots placed one-by-one "
          f"(each survey-grade + independently verified before the next)")

    # --- Relative growth (client's FMBS_STONES_MATCH), also ONE verified commit per round ---
    # Seat a still-unplaced plot against a committed neighbour by their SHARED FMB corners
    # (relative_club.propagate_geometric), placed RIGIDLY (scale=1, rule 2). A relative tie is
    # trusted only when it ALSO lands at the plot's OWN cadastre seat (a THIRD source) -- the
    # >=3-shared-corner + tiling gate alone admits chance ties 150-1100 m off (measured).
    def _anchor_cp(sv):
        pl = placed[sv]; m1 = m1s[sv]; pos = m1.stone_positions()
        adj = pos @ pl["R"].T + pl["t"]                    # scale=1 rigid, ALL stones
        return CandidatePlacement(method="m3_anchor", R=pl["R"], s=1.0, t=pl["t"], adjusted=adj,
                                  corner_ring=list(m1.outer_stone_indices), passes_gate=True,
                                  scale=1.0)

    n_rel = 0
    while True:
        anchor_cps = {sv: _anchor_cp(sv) for sv in placed}
        tile_fps = list(placed_fp.values())
        best = None                                        # (sv, rp, fp, score)
        for sv, m1 in m1s.items():
            if sv in placed:
                continue
            seat = seeds.get(sv)
            if seat is None:                               # cannot verify a relative tie -> skip
                continue
            for asv, acp in anchor_cps.items():
                cand = propagate_geometric(m1, acp, m1s[asv], tile_fps)
                if cand is None:
                    continue
                rp = _place_relative(m1, cand, acp.corner_points())
                if rp is None:
                    continue
                pc = rp["ring"].mean(axis=0)
                cr = float(np.hypot(pc[0] - seat[0], pc[1] - seat[1]))
                if cr > M3_CORROB_TOL_M:                   # VERIFY vs the independent cadastre seat
                    continue
                fp = _fp(rp["ring"])
                if fp is None or _overlaps(fp, placed_fp):
                    continue
                score = rp["n_matched"] - 0.02 * cr
                if best is None or score > best[3]:
                    rp2 = dict(rp); rp2["relative_to"] = asv; rp2["corrob"] = cr
                    best = (sv, rp2, fp, score)
        if best is None:
            break
        sv, rp, fp, _s = best
        placed[sv], placed_fp[sv] = rp, fp                 # commit ONE verified relative tie
        n_rel += 1
    if n_rel:
        print(f"[3b/4] relative stone-match: +{n_rel} plot(s) seated one-by-one by shared FMB "
              f"corners with a confirmed neighbour (client FMBS_STONES_MATCH, cadastre-verified)")

    # --- Honest first-class dispositions + the three deliverables (rule 2: scale-locked) ---
    placements = []
    for sv, m1 in m1s.items():
        n = len(m1.outer_stone_indices)
        pl = placed.get(sv)
        if pl is not None and pl.get("relative_to"):
            # Placed by the client's shared-stone match to a CONFIRMED neighbour. It already
            # passed the client's own 0-FP bar (>=3 coincident FMB corners + tiling), so it is
            # judged on THAT provenance, not re-gated by the solo-stone residual bar. A DISTINCT
            # tier: confidently LOCATED via a confirmed neighbour, with FMB-fidelity extent at
            # the non-shared corners -- never conflated with the dual-confirmed survey-grade set.
            res = pl["residuals"]
            med = float(np.median(res)) if len(res) else float("nan")
            mx = float(np.max(res)) if len(res) else float("nan")
            cr = pl.get("corrob", float("nan"))
            placements.append(M3Placement(
                survey_number=sv, disposition="ACCEPT_RELATIVE", R=pl["R"], t=pl["t"],
                s_fitted=pl["s_fitted"], ring_utm=pl["ring"], n_matched=pl["n_matched"],
                n_corners=n, median_residual_m=med, max_residual_m=mx,
                cadastre_corrob_m=(float("nan") if cr is None else cr),
                note=f"relative stone-match to confirmed {pl['relative_to']} "
                     f"({pl['n_matched']} shared corners, rigid); cadastre-verified "
                     f"{cr:.0f}m, FMB-fidelity extent"))
        elif pl is not None:
            # COMMITTED by the sequential verified growth: survey-grade median AND (independent
            # cadastre agreement OR dense clean self-match) AND it tiles -- multiple independent
            # sources agreed BEFORE it was committed. -> ACCEPT (accuracy never relaxed).
            res = pl["residuals"]
            med = float(np.median(res)) if len(res) else float("nan")
            mx = float(np.max(res)) if len(res) else float("nan")
            corrob = pl.get("corrob")
            src = ("cadastre-corroborated {:.1f}m (2 independent sources)".format(corrob)
                   if corrob is not None and corrob <= M3_CORROB_TOL_M else "dense self-match")
            placements.append(M3Placement(
                survey_number=sv, disposition="ACCEPT", R=pl["R"], t=pl["t"],
                s_fitted=pl["s_fitted"], ring_utm=pl["ring"], n_matched=pl["n_matched"],
                n_corners=n, median_residual_m=med, max_residual_m=mx,
                cadastre_corrob_m=(float("nan") if corrob is None else corrob),
                note=f"n={pl['n_matched']}/{n} med={med:.2f}m max={mx:.2f}m "
                     f"(survey-grade, verified: {src})"))
        else:
            # NOT committed. If its best direct attempt still LANDS at its cadastre seat it is
            # correctly located but sub-survey-grade / lost a tiling conflict -> REVIEW (geometry
            # shown). Otherwise (decoy / no seat) -> NEEDS_GPS, no geometry drawn (never a fake).
            pl2 = direct.get(sv)
            corrob = pl2.get("corrob") if pl2 else None
            if pl2 is not None and corrob is not None and corrob <= M3_CORROB_TOL_M:
                res = pl2["residuals"]
                med = float(np.median(res)) if len(res) else float("nan")
                mx = float(np.max(res)) if len(res) else float("nan")
                placements.append(M3Placement(
                    survey_number=sv, disposition="REVIEW", R=pl2["R"], t=pl2["t"],
                    s_fitted=pl2["s_fitted"], ring_utm=pl2["ring"], n_matched=pl2["n_matched"],
                    n_corners=n, median_residual_m=med, max_residual_m=mx,
                    cadastre_corrob_m=corrob,
                    note=f"n={pl2['n_matched']}/{n} med={med:.2f}m cadastre-agrees {corrob:.0f}m "
                         f"but sub-survey-grade / tiling conflict -> confirm"))
            else:
                disp, note = classify(0, n, float("nan"), float("nan"), float("nan"),
                                      tiles=True, window_has_stones=True)
                placements.append(M3Placement(survey_number=sv, disposition=disp,
                                              n_corners=n, note=note))

    for p in placements:
        p.village = village                                # tag for the combined district output
    counts: dict[str, int] = {}
    for p in placements:
        counts[p.disposition] = counts.get(p.disposition, 0) + 1
    print(f"[4/4] {village}: {counts.get('ACCEPT', 0)} ACCEPT (survey-grade) + "
          f"{counts.get('ACCEPT_RELATIVE', 0)} ACCEPT_RELATIVE (shared-stone) = "
          f"{counts.get('ACCEPT', 0) + counts.get('ACCEPT_RELATIVE', 0)}/{len(placements)} "
          f"clubbed | {dict(counts)}")
    return placements, spos


def main() -> None:
    """M3 = ONE clubbed output. Takes 1+ villages; each is placed by ``place_village`` against the
    SAME surveyor stone cloud, and ALL placements are written to a SINGLE DXF/report/overlay (the
    surveyor RAW DATA is one cloud spanning the district). Per-village runs stay backward-compatible.
    Usage: python run_m3.py <VILLAGE> [VILLAGE ...] --surveyor "<RAW DATA.dxf>" [--district Erode]"""
    a = sys.argv[1:]
    villages = []
    for tok in a:
        if tok.startswith("--"):
            break
        villages.append(tok)
    surveyor = a[a.index("--surveyor") + 1] if "--surveyor" in a else None
    district = a[a.index("--district") + 1] if "--district" in a else "Erode"
    taluk = a[a.index("--taluk") + 1] if "--taluk" in a else ""
    if not surveyor or not villages:
        raise SystemExit("usage: python run_m3.py <VILLAGE> [VILLAGE ...] "
                         "--surveyor <RAW DATA.dxf> [--district Erode]")

    all_pl, all_stones = [], []
    for v in villages:
        placements, spos = place_village(v, surveyor, district, taluk)
        all_pl += placements
        if spos is not None and len(spos):
            all_stones.append(spos)
    stones = np.vstack(all_stones) if all_stones else np.empty((0, 2))

    # SINGLE M3 OUTPUT: every village's plots clubbed onto the shared surveyor stones in ONE DXF.
    single = len(villages) == 1
    outdir = Path(f"output/{villages[0]}/m3") if single else Path("output/M3_CLUBBED")
    title = villages[0] if single else "+".join(villages)
    write_dxf(all_pl, outdir / "clubbed_village.dxf", crs=CRS)
    write_overlay(all_pl, stones, outdir / "qa_overlay.png", village=title)
    write_report(all_pl, outdir / "m3_report.json", village=title)

    counts: dict[str, int] = {}
    for p in all_pl:
        counts[p.disposition] = counts.get(p.disposition, 0) + 1
    acc = sorted(f"{p.village}:{p.survey_number}" for p in all_pl if p.disposition == "ACCEPT")
    rel = sorted(f"{p.village}:{p.survey_number}" for p in all_pl
                 if p.disposition == "ACCEPT_RELATIVE")
    n_ok = counts.get("ACCEPT", 0) + counts.get("ACCEPT_RELATIVE", 0)
    print(f"\n================ M3 SINGLE CLUBBED OUTPUT: {title} ================")
    print(f"{n_ok}/{len(all_pl)} plots clubbed onto the surveyor stone cloud "
          f"({counts.get('ACCEPT', 0)} survey-grade + {counts.get('ACCEPT_RELATIVE', 0)} "
          f"shared-stone), ALL cadastre-verified | dispositions {dict(counts)}")
    print(f"  ACCEPT (survey-grade)  : {acc}")
    print(f"  ACCEPT_RELATIVE        : {rel}")
    print(f"  ONE DXF -> {outdir}/clubbed_village.dxf   (+ qa_overlay.png, m3_report.json)")


if __name__ == "__main__":
    main()
